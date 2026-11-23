#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.metrics.bootstrap import point_values, request_outcomes

RULES = ("best_quality", "zero_violation")
SCORES = {"cascade": ("cascade_score", "geometric-mean token probability"),
          "automix": ("automix_score", "share of 'Correct' self-checks")}


def load_run(task_dir: str) -> dict:
    table = json.loads((Path(task_dir) / "table.json").read_text())
    run = ROOT / "runs" / table["rows"][0]["run"]
    results = {r["id"]: r for r in map(json.loads, open(run / "results.jsonl"))}
    return {"run": run.name, "edge": table["edge"], "results": results, "outcomes": request_outcomes(run)}


def route(ids, scores, t, edge, redact) -> list[dict]:
    rows = []
    for i in ids:
        sent = scores[i] is not None and scores[i] < t and not redact["results"][i]["notes"].get("fallback")
        rows.append(dict(redact["outcomes"][i] if sent else edge["outcomes"][i], escalated=int(sent)))
    return rows


def point(ids, scores, t, edge, redact) -> dict:
    rows = route(ids, scores, t, edge, redact)
    return {"threshold": t, "escalated": sum(r["escalated"] for r in rows) / len(rows)} | point_values(rows)


def curve(ids, scores, edge, redact) -> list[dict]:
    levels = {s for s in scores.values() if s is not None}
    return [point(ids, scores, t, edge, redact) for t in sorted({0.0, *levels, 1.01})]


def select(ids, scores, edge, redact) -> dict:
    levels = sorted({s for s in scores.values() if s is not None})
    cuts = [0.0] + [(a + b) / 2 for a, b in zip(levels, levels[1:])] + [levels[-1] + 0.01]
    pts = [point(ids, scores, t, edge, redact) for t in cuts]
    def best(cands):
        return max(cands, key=lambda p: (p["at_least_as_good"] or 0.0, -p["escalated"]))
    return {"best_quality": best(pts), "zero_violation": best([p for p in pts if p["hard_violation"] == 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--edge-only", required=True)
    ap.add_argument("--redact", required=True)
    ap.add_argument("--fixed", help="selected.json from the development split, applied here unchanged")
    ap.add_argument("--register", help="record the thresholds picked here in this file (configs/baseline_thresholds.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scored = {s["id"]: s for s in map(json.loads, open(Path(args.scores) / "scores.jsonl"))}
    edge, redact = load_run(args.edge_only), load_run(args.redact)
    ids = [i for i in scored if i in edge["results"] and i in redact["results"]]
    same = sum(scored[i]["answer"].strip() == edge["results"][i]["answer"].strip() for i in ids) / len(ids)
    scores = {name: {i: scored[i][key] for i in ids} for name, (key, _) in SCORES.items()}
    out_dir = Path(args.out)
    out = {"scores": args.scores, "edge": edge["edge"], "edge_only_run": edge["run"], "redact_send_run": redact["run"],
           "requests": len(ids), "edge_answer_identical": same,
           **{name: curve(ids, scores[name], edge, redact) for name in SCORES}}
    selected = {name: select(ids, scores[name], edge, redact) for name in SCORES}
    fixed = {}
    if args.fixed:
        chosen = json.loads(Path(args.fixed).read_text())
        (out_dir / "fixed").mkdir(parents=True, exist_ok=True)
        for name in SCORES:
            for rule in RULES:
                t = chosen[name][rule]["threshold"]
                fixed[f"{name}-{rule}"] = point(ids, scores[name], t, edge, redact)
                rows = route(ids, scores[name], t, edge, redact)
                (out_dir / "fixed" / f"{name}-{rule}.json").write_text(json.dumps(
                    {"edge": edge["edge"], "system": name, "rule": rule, "threshold": t, "from": args.fixed,
                     "outcomes": dict(zip(ids, rows))}, indent=1))
        out["fixed"] = fixed
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "curves.json").write_text(json.dumps(out, indent=1))
    (out_dir / "selected.json").write_text(json.dumps(selected, indent=1))
    if args.register:
        reg = Path(args.register)
        table = json.loads(reg.read_text()) if reg.exists() else {}
        table[edge["edge"]] = {
            "picked_on": f"{len(ids)} requests of {json.loads((Path(args.edge_only) / 'table.json').read_text()).get('data', 'pupa_tnb')}"
                         f" ({args.scores}, {edge['run']}, {redact['run']}), {time.strftime('%Y-%m-%d')}",
            **{name: {rule: selected[name][rule]["threshold"] for rule in RULES} for name in SCORES}}
        reg.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")

    def pct(x):
        return "–" if x is None else f"{100 * x:.1f}"
    header = ["| {} | Sent to cloud (%) | ≥ reference (%) | Hard violations (%) | Local-only spans leaked (%) "
              "| Attacker (%) | Cloud tokens in/out |", "|---|---|---|---|---|---|---|"]

    def line(label, p):
        return (f"| {label} | {pct(p['escalated'])} | {pct(p['at_least_as_good'])} | {pct(p['hard_violation'])} | "
                f"{pct(p['span_leak_local_only'])} | {pct(p['attacker_reconstruction'])} | "
                f"{p['cloud_in_tokens']:.0f} / {p['cloud_out_tokens']:.0f} |")
    lines = [f"# AutoMix and cascade at every threshold ({len(ids)} requests, edge {edge['edge']})", "",
             f"Edge-only run `{edge['run']}`, redact-and-send run `{redact['run']}`. Edge answers re-created by the "
             f"scoring job are identical to the edge-only run's for {100 * same:.0f}% of requests. With every request "
             "escalated, the curve differs from the redact-and-send run only where the guard denied: AutoMix and cascade "
             "keep their draft (the edge-only answer), while redact-and-send makes a new edge call after the labeler "
             "call, which rarely comes out identical.", ""]
    lines += ["## Thresholds picked on these requests (selected.json)", "", header[0].format("System, rule (threshold)"),
              header[1]]
    lines += [line(f"{name}, {rule} ({selected[name][rule]['threshold']:.3f})", selected[name][rule])
              for name in SCORES for rule in RULES]
    lines.append("")
    if fixed:
        lines += [f"## Thresholds from `{args.fixed}`, applied unchanged", "", header[0].format("System, rule (threshold)"),
                  header[1]]
        lines += [line(f"{k.replace('-', ', ', 1)} ({p['threshold']:.3f})", p) for k, p in fixed.items()]
        lines.append("")
    for name, (_, desc) in SCORES.items():
        lines += [f"## {name.capitalize()} (score: {desc})", "", header[0].format("Threshold"), header[1]]
        seen = set()
        for p in out[name]:
            key = round(p["escalated"], 4)
            if key in seen:
                continue
            seen.add(key)
            lines.append(line(f"{p['threshold']:.3f}", p))
        lines.append("")
    (out_dir / "curves.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
