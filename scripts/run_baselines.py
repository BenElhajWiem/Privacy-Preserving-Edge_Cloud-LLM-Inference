#!/usr/bin/env python
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))
from envfile import load_env

SYSTEMS = {
    "edge_only": {},
    "cloud_full": {},
    "redact_send": {},
    "automix": {"k": 4},
    "cascade": {},
    "papillon": {"prompts": "llama_32_1b"},
    "prism": {},
}
THRESHOLDS = ROOT / "configs/baseline_thresholds.json"
METRICS = ["at_least_as_good", "hard_violation", "hard_violation_strict", "span_leak_local_only",
           "abstraction_compliance", "attacker_reconstruction", "attacker_prior", "cloud_calls", "cloud_in_tokens",
           "cloud_out_tokens", "edge_s", "wall_s"]


def run_systems(args) -> None:
    load_env()
    from gapdeleg.build.io import request_ids
    from gapdeleg.experiment import run

    ids = request_ids(args.data, args.split, args.sample, args.seed)
    if args.ids_file:
        keep = set(Path(args.ids_file).read_text().split())
        ids = [i for i in ids if i in keep]
    out = Path(args.out) if args.out else ROOT / "runs" / f"baselines-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    edge_tag = args.edge.replace(":", "-")
    rows = []
    for name in args.systems.split(","):
        system_args = dict(SYSTEMS[name])
        label = name
        if name == "papillon":
            system_args["prompts"] = args.papillon_prompts
        if name == "prism":
            system_args["seed"] = args.prism_seed
            label = f"prism-s{args.prism_seed}"
        if name in ("automix", "cascade"):
            tuned = json.loads(THRESHOLDS.read_text()).get(args.edge, {}).get(name)
            if not tuned:
                sys.exit(f"no {name} threshold for {args.edge} in {THRESHOLDS}: tune it on the development split first")
            system_args["threshold"] = tuned[args.threshold_rule]
            label = f"{name}-{args.threshold_rule}"
        if args.tag:
            label = f"{label}-{args.tag}"
        cfg = {"name": f"baseline_{label}_{edge_tag}", "system": name, "system_args": system_args,
               "data": {"dataset": "processed", "name": args.data, "ids": ids},
               "edge": {"backend": "ollama", "model": args.edge, "num_ctx": 16384},
               "cloud": {"backend": "ollama", "model": args.cloud} if args.cloud_backend == "ollama" else {"backend": "anthropic", "model": args.cloud, "effort": "low"},
               "cloud_cache": args.cloud_cache,
               "labeler": ["detector", "edge"] if name in ("redact_send", "automix", "cascade") else [],
               "judge": None if args.no_judge else {"backend": "openai"},
               "attacker": None if args.no_attacker else {"backend": "anthropic", "model": args.cloud, "effort": "low"},
               "gen": {"edge_max_tokens": 1024, "cloud_max_tokens": 2048}}
        t = time.time()
        row = {"system": name, "edge": args.edge}
        if name == "papillon":
            row["papillon_prompts"] = args.papillon_prompts
        if name == "prism":
            row["prism_seed"] = args.prism_seed
        if name in ("automix", "cascade"):
            row.update(threshold_rule=args.threshold_rule, threshold=system_args["threshold"])
        if args.tag:
            row["tag"] = args.tag
        sampler = OllamaMemory(cfg["edge"].get("host")) if args.measure_memory else None
        try:
            if sampler:
                sampler.start()
            rd = run(cfg)
            if sampler:
                row["memory"] = sampler.stop()
                s_path = rd.path / "summary.json"
                s_data = json.loads(s_path.read_text())
                s_data["memory"] = row["memory"]
                s_path.write_text(json.dumps(s_data, indent=1))
            s = json.loads((rd.path / "summary.json").read_text())
            recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
            errors = sum(bool(r["error"]) for r in recs)
            empty = sum(not r["answer"] for r in recs)
            row.update({"run": rd.id, "status": "ok" if errors < len(recs) / 2 else "mostly failed",
                        "n": len(recs), "errors": errors, "empty_answers": empty,
                        "at_least_as_good": s.get("quality", {}).get("at_least_as_good"),
                        "exact_match": s.get("exact_match"),
                        "cloud_spend_usd": s.get("cloud_spend_usd"),
                        "verdicts": s.get("quality", {}).get("verdicts"),
                        "judge_errors": s.get("quality", {}).get("judge_errors", 0),
                        "hard_violation": s["leakage"]["hard_violation_rate"],
                        "hard_violation_strict": s["leakage"]["hard_violation_rate_strict"],
                        "span_leak_local_only": s["leakage"]["per_label"].get("local-only", {}).get("rate"),
                        "abstraction_compliance": s["leakage"]["abstraction_compliance"],
                        "attacker_reconstruction": s.get("attacker", {}).get("reconstruction_rate"),
                        "attacker_prior": s.get("attacker", {}).get("prior_rate"),
                        "cloud_calls": s["cost"]["cloud_calls"]["mean"],
                        "cloud_in_tokens": s["cost"]["cloud_in_tokens"]["mean"],
                        "cloud_out_tokens": s["cost"]["cloud_out_tokens"]["mean"],
                        "edge_s": s["cost"]["edge_latency_s"]["mean"], "wall_s": s["cost"]["wall_s"]["mean"],
                        "minutes": round((time.time() - t) / 60, 1)})
        except Exception as e:
            if sampler and not sampler._stop.is_set():
                sampler.stop()
            row.update({"status": f"failed: {type(e).__name__}: {e}"[:300], "minutes": round((time.time() - t) / 60, 1)})
            (out / f"{label}_error.txt").write_text(traceback.format_exc())
        rows.append(row)
        (out / "table.json").write_text(json.dumps({"ids": ids, "data": args.data, "split": args.split,
                                                    "seed": args.seed, "edge": args.edge,
                                                    "cloud": args.cloud, "rows": rows}, indent=1))
        print(json.dumps(row), flush=True)
    write_table(out, f"# Baselines on {data_label(args.data, args.split)} ({len(ids)} items, seed {args.seed}); edge {args.edge}, "
                     f"cloud {args.cloud}", rows)


class OllamaMemory:

    def __init__(self, host: str | None = None, every_s: float = 0.5):
        import os
        self.host = host or os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
        self.every_s, self.peak_rss, self.peak_reported, self._stop = every_s, 0, 0, None

    def _sample(self):
        import psutil
        import urllib.request
        n = 0
        while not self._stop.is_set():
            rss = 0
            for pr in psutil.process_iter(["name", "exe"]):
                try:
                    if "ollama" in ((pr.info["name"] or "") + (pr.info["exe"] or "")).lower():
                        rss += pr.memory_info().rss
                except psutil.Error:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            if n % 10 == 0:
                try:
                    with urllib.request.urlopen(f"{self.host}/api/ps", timeout=2) as r:
                        self.peak_reported = max([self.peak_reported] + [m.get("size", 0) for m in json.load(r)["models"]])
                except OSError:
                    pass
            n += 1
            self._stop.wait(self.every_s)

    def start(self):
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {"peak_ollama_rss_gb": round(self.peak_rss / 2**30, 3),
                "ollama_reported_model_gb": round(self.peak_reported / 2**30, 3)}


def data_label(data: str, split: str | None) -> str:
    return {"pupa_tnb": "PUPA-TNB", "pupa_new": "PUPA-New"}.get(data, data) + (f" {split}" if split else "")


def collect(args) -> None:
    tables = [json.loads((Path(d) / "table.json").read_text()) for d in args.collect if (Path(d) / "table.json").exists()]
    if not tables:
        sys.exit("no table.json found in the given directories")
    rows = [dict(r, edge=r.get("edge", t["edge"])) for t in tables for r in t["rows"]]
    order = list(SYSTEMS)
    rows.sort(key=lambda r: (r["edge"], order.index(r["system"]), r.get("prism_seed", 0)))
    sections, merged = [], []
    for edge in sorted({r["edge"] for r in rows}):
        group = [r for r in rows if r["edge"] == edge]
        prism = [r for r in group if r["system"] == "prism" and "n" in r]
        if len(prism) > 1:
            mean = {"system": "prism", "edge": edge, "prism_seed": "mean", "status": "ok", "n": prism[0]["n"],
                    "errors": sum(r["errors"] for r in prism), "empty_answers": sum(r["empty_answers"] for r in prism)}
            for m in METRICS:
                vals = [r[m] for r in prism if r.get(m) is not None]
                mean[m] = sum(vals) / len(vals) if vals else None
            group.append(mean)
        merged += group
        t = next(t for t in tables if t["edge"] == edge)
        sections.append(write_table(None, f"## Edge {edge} ({len(t['ids'])} {data_label(t.get('data', 'pupa_tnb'), t.get('split'))} "
                                          f"items, seed {t.get('seed', 0)}, "
                                          f"cloud {t['cloud']})", group))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps({"ids": tables[0]["ids"], "rows": merged}, indent=1))
    (out / "table.md").write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


def write_table(out: Path | None, title: str, rows: list[dict]) -> str:
    def f(x, pct=True):
        return "–" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.1f}")
    lines = [title, "", "| System | ≥ reference (%) | Hard violations (%) | strict (%) | Local-only spans leaked (%) | "
             "Abstraction compliance (%) | Attacker / prior (%) | Cloud calls | Cloud tokens in/out | Edge s | Status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = r["system"] if "prism_seed" not in r else f"prism ({'mean of seeds' if r['prism_seed'] == 'mean' else 'seed ' + str(r['prism_seed'])})"
        if "threshold_rule" in r:
            name = f"{r['system']} ({r['threshold_rule'].replace('_', ' ')}, threshold {r['threshold']:.3f})"
        if r.get("tag"):
            name += f" [{r['tag']}]"
        if "n" not in r:
            lines.append(f"| {name} | | | | | | | | | | {r['status']} |")
            continue
        lines.append(f"| {name} | {f(r['at_least_as_good'])} | {f(r['hard_violation'])} | "
                     f"{f(r['hard_violation_strict'])} | {f(r['span_leak_local_only'])} | "
                     f"{f(r['abstraction_compliance'])} | {f(r['attacker_reconstruction'])} / {f(r['attacker_prior'])} | "
                     f"{r['cloud_calls']:.2f} | {r['cloud_in_tokens']:.0f} / {r['cloud_out_tokens']:.0f} | "
                     f"{r['edge_s']:.1f} | {r['status']} ({r['errors']} errors, {r['empty_answers']} empty"
                     f"{', %d unjudged' % r['judge_errors'] if r.get('judge_errors') else ''}) |")
    text = "\n".join(lines)
    if out is not None:
        (out / "table.md").write_text(text + "\n")
        print(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pupa_tnb", help="processed dataset, e.g. pupa_tnb or pupa_new")
    ap.add_argument("--split", help="one split of data/processed/splits.json, e.g. dev")
    ap.add_argument("--sample", type=int, default=80, help="seeded random sample size; 0 keeps every request")
    ap.add_argument("--ids-file", help="keep only these request ids (one per line), e.g. configs/mac_cost_ids.txt")
    ap.add_argument("--measure-memory", action="store_true",
                    help="record the peak memory of the local Ollama processes (on-device cost runs)")
    ap.add_argument("--no-attacker", action="store_true", help="skip the attacker (e.g. on the development split)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the judge: on-device cost runs, or judge later in one batch with scripts/judge_runs.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--edge", default="qwen3:1.7b")
    ap.add_argument("--cloud", default="claude-sonnet-5-5")
    ap.add_argument("--cloud-backend", choices=["anthropic", "ollama"], default="anthropic",
                    help="ollama: use a local Ollama model (--cloud <tag>) as the cloud, no API key needed")
    ap.add_argument("--prism-seed", type=int, default=0)
    ap.add_argument("--papillon-prompts", default=SYSTEMS["papillon"]["prompts"],
                    help="PAPILLON's published prompts to use, e.g. llama_32_3b for Llama-3.2-3B")
    ap.add_argument("--threshold-rule", choices=["best_quality", "zero_violation"], default="best_quality",
                    help="which development-split threshold AutoMix and cascade use")
    ap.add_argument("--cloud-cache", default="runs/cache",
                    help="cache for the system's own cloud calls (judge and attacker keep runs/cache); a fresh "
                         "directory gives uncached cloud answers, as in the cloud-variance check")
    ap.add_argument("--tag", help="added to the run name, e.g. r1 for a repeat")
    ap.add_argument("--out", help="output directory (default: runs/baselines-<time>)")
    ap.add_argument("--collect", nargs="+", metavar="DIR", help="merge these runs' table.json files into --out")
    args = ap.parse_args()
    if args.collect:
        if not args.out:
            ap.error("--collect needs --out")
        collect(args)
    else:
        run_systems(args)


if __name__ == "__main__":
    main()
