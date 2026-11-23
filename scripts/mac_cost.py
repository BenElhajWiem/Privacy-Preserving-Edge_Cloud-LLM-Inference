#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(run: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())}


def steps(recs) -> list[tuple[float, float, float]]:
    return [(s["in"] or 0, s["out"] or 0, s["latency_s"] or 0.0) for r in recs for s in r["edge_steps"]]


def predict(coef, rec) -> float:
    return sum(coef[0] + coef[1] * (s["in"] or 0) + coef[2] * (s["out"] or 0) for s in rec["edge_steps"])


def key(row) -> tuple:
    return row["system"], row["edge"], row.get("threshold_rule"), row.get("papillon_prompts")


GAP_LOOP_STEP = re.compile(r"^(gap_report_r\d+|rewrite_r\d+|guard_cloud_r([2-9]|\d\d+)(_.*)?)$")
METHOD_DEV = {"privfin": "famb/claude/formula-alone/rows/py-clean-ex__privfin_dev",
              "privmed": "famb/claude/refs/rows/v0-gate-open__privmed_dev"}


def is_method(row) -> bool:
    return row["system"].startswith("method_")


def run_records(path: str) -> tuple[dict[str, dict], str]:
    d = json.loads((ROOT / path).read_text())
    run = d["rows"][0]["run"] if "rows" in d else d["run"]
    return load(ROOT / "runs" / run), run


def method_reference(row) -> tuple[dict, dict, str]:
    s = row["edge"].replace(":", "-")
    if row["system"] == "method_pupa":
        op, op_run = run_records(f"runs/method-pilot/open-{s}/table.json")
        cl, cl_run = run_records(f"runs/method-pilot/closed-{s}/table.json")
        t = row["threshold"]
        recs = {i: {"edge_steps": [x for x in (o if o["notes"]["confidence"] < t else cl[i])["edge_steps"]
                                   if not GAP_LOOP_STEP.match(x["step"])]} for i, o in op.items()}
        return recs, recs, f"{op_run} + {cl_run} at t={t:.3f}, no gap loop"
    fam = row["system"].split("_", 1)[1]
    dev, dev_run = run_records(f"runs/{METHOD_DEV[fam]}__{s}.json")
    test, test_run = run_records(f"runs/famb-test/method/rows/{fam}_test__{s}.json")
    return dev, test, f"{dev_run} (sample) + {test_run} (estimate)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run_baselines.py output folders of the Mac runs")
    ap.add_argument("--cluster-table", default="runs/baselines-final/table.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cluster = {}
    for row in json.loads((ROOT / args.cluster_table).read_text())["rows"]:
        if "run" in row and row.get("prism_seed", 0) in (0, None):
            cluster.setdefault(key(row), row)
    mac = []
    for d in args.dirs:
        p = Path(d) / "table.json"
        if p.exists():
            row = json.loads(p.read_text())["rows"][0]
            if "run" in row:
                summary = json.loads((ROOT / "runs" / row["run"] / "summary.json").read_text())
                mac.append({"row": row, "recs": load(ROOT / "runs" / row["run"]), "memory": summary.get("memory", {})})

    fits = {}
    for model in sorted({m["row"]["edge"] for m in mac}):
        pts = np.array([s for m in mac if m["row"]["edge"] == model and not is_method(m["row"])
                        for s in steps(m["recs"].values())], dtype=float)
        x = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1]])
        coef, *_ = np.linalg.lstsq(x, pts[:, 2], rcond=None)
        if coef[0] < 0:
            coef = np.concatenate([[0.0], np.linalg.lstsq(x[:, 1:], pts[:, 2], rcond=None)[0]])
        resid = pts[:, 2] - x @ coef
        r2 = 1 - (resid ** 2).sum() / ((pts[:, 2] - pts[:, 2].mean()) ** 2).sum()
        fits[model] = {"coef": coef.tolist(), "steps": len(pts), "r2": float(r2),
                       "prompt_tok_per_s": float(1 / coef[1]) if coef[1] > 0 else None,
                       "write_tok_per_s": float(1 / coef[2]) if coef[2] > 0 else None}

    rows = []
    for m in mac:
        r, recs = m["row"], m["recs"]
        coef = fits[r["edge"]]["coef"]
        ids = list(recs)
        measured = float(np.mean([x["cost"]["edge_latency_s"] for x in recs.values()]))
        wall = float(np.mean([x["cost"]["wall_s"] for x in recs.values()]))
        sp = None if is_method(r) else cluster.get(key(r))
        out = {"system": r["system"], "edge": r["edge"], "rule": r.get("threshold_rule"), "requests": len(ids),
               "measured_edge_s": measured, "measured_wall_s": wall, **m["memory"]}
        if is_method(r):
            sample, full, label = method_reference(r)
            same = [i for i in ids if i in sample]
            out.update(cluster_run=label, sample_matched=len(same), cloud_calls=r.get("cloud_calls"),
                       predicted_edge_s_sample=float(np.mean([predict(coef, sample[i]) for i in same])),
                       predicted_edge_s_all=float(np.mean([predict(coef, x) for x in full.values()])),
                       all_requests=len(full))
            out["estimated_edge_s_all"] = measured * out["predicted_edge_s_all"] / out["predicted_edge_s_sample"]
        elif sp:
            srec = load(ROOT / "runs" / sp["run"])
            same = [i for i in ids if i in srec]
            out.update(cluster_run=sp["run"],
                       predicted_edge_s_sample=float(np.mean([predict(coef, srec[i]) for i in same])),
                       predicted_edge_s_all=float(np.mean([predict(coef, x) for x in srec.values()])),
                       all_requests=len(srec))
            out["estimated_edge_s_all"] = measured * out["predicted_edge_s_all"] / out["predicted_edge_s_sample"]
        rows.append(out)

    mem_path = args.out / "memory.json"
    memory = json.loads(mem_path.read_text())["models"] if mem_path.exists() else {}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cost.json").write_text(json.dumps({"fits": fits, "memory": memory, "rows": rows}, indent=1))

    def f(x, d=1):
        return "–" if x is None else f"{x:.{d}f}"
    lines = ["# On-device cost on the Mac (Apple M2, 16 GB), sampled and extrapolated", "",
             "Per edge model, seconds per on-device step = fixed + prompt tokens / prompt rate + written tokens / "
             "writing rate, fitted on every step of the Mac runs:", "",
             "| Edge model | Steps | Fixed (s) | Prompt (tok/s) | Writing (tok/s) | R² | Peak memory, alone (GB) | "
             "Ollama's allocation (GB) |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {mdl} | {v['steps']} | {f(v['coef'][0], 2)} | {f(v['prompt_tok_per_s'], 0)} | "
              f"{f(v['write_tok_per_s'], 1)} | {f(v['r2'], 3)} | {f(memory.get(mdl, {}).get('peak_ollama_rss_gb'), 2)} | "
              f"{f(memory.get(mdl, {}).get('ollama_reported_model_gb'), 2)} |" for mdl, v in fits.items()]
    lines += ["", "Per request: on-device seconds measured on the sample; the same predicted from the cluster run's "
              "token counts (the check); and the estimate for all pilot requests (measured, scaled for the "
              "sample's mix). Peak memory during the run can include a previous model still loaded.", "",
              "| Edge model | System | Measured, sample (s) | Check: predicted, sample (s) | Estimate, all (s) | "
              "Wall clock, sample (s) | Peak memory during the run (GB) |", "|---|---|---|---|---|---|---|"]
    for o in sorted(rows, key=lambda o: (o["edge"], o["system"])):
        name = o["system"] + (f" ({o['rule'].replace('_', ' ')})" if o.get("rule") else "")
        lines.append(f"| {o['edge']} | {name} | {f(o['measured_edge_s'])} ({o['requests']}) | "
                     f"{f(o.get('predicted_edge_s_sample'))} | {f(o.get('estimated_edge_s_all'))} "
                     f"({o.get('all_requests', '–')}) | {f(o['measured_wall_s'])} | {f(o.get('peak_ollama_rss_gb'), 2)} |")
    (args.out / "cost.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
