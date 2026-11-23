#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.metrics.bootstrap import point_values, request_outcomes

RULES = ("best_quality", "zero_violation")


def load_run(task_dir: str) -> dict:
    table = json.loads((Path(task_dir) / "table.json").read_text())
    run = ROOT / "runs" / table["rows"][0]["run"]
    results = {r["id"]: r for r in map(json.loads, open(run / "results.jsonl"))}
    return {"run": run.name, "edge": table["edge"], "results": results, "outcomes": request_outcomes(run)}


def route(ids, conf, t, open_, closed) -> list[dict]:
    return [dict(open_["outcomes"][i] if conf[i] < t else closed["outcomes"][i], escalated=int(conf[i] < t))
            for i in ids]


def point(ids, conf, t, open_, closed) -> dict:
    rows = route(ids, conf, t, open_, closed)
    return {"threshold": t, "escalated": sum(r["escalated"] for r in rows) / len(rows)} | point_values(rows)


def select(ids, conf, open_, closed) -> dict:
    levels = sorted(set(conf.values()))
    cuts = [0.0] + [(a + b) / 2 for a, b in zip(levels, levels[1:])] + [levels[-1] + 0.01]
    pts = [point(ids, conf, t, open_, closed) for t in cuts]

    def best(cands):
        return max(cands, key=lambda p: (p["at_least_as_good"] or 0.0, -p["escalated"])) if cands else None
    return {"best_quality": best(pts), "zero_violation": best([p for p in pts if p["hard_violation"] == 0])}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", required=True)
    ap.add_argument("--closed", required=True)
    ap.add_argument("--fixed", help="selected.json from the development split, applied here unchanged")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    open_, closed = load_run(args.open), load_run(args.closed)
    ids = [i for i in open_["results"] if i in closed["results"]]
    conf = {i: open_["results"][i]["notes"].get("confidence", 0.0) for i in ids}
    diff = [abs(conf[i] - closed["results"][i]["notes"].get("confidence", 0.0)) for i in ids]
    same = sum(d < 1e-6 for d in diff) / len(ids)
    out = {"edge": open_["edge"], "open_run": open_["run"], "closed_run": closed["run"], "requests": len(ids),
           "confidence_identical": same, "confidence_max_difference": max(diff),
           "curve": [point(ids, conf, t, open_, closed) for t in sorted({0.0, *conf.values(), 1.01})]}
    selected = select(ids, conf, open_, closed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fixed:
        chosen = json.loads(Path(args.fixed).read_text())
        (out_dir / "fixed").mkdir(exist_ok=True)
        out["fixed"] = {}
        for rule in RULES:
            if not chosen.get(rule):
                continue
            t = chosen[rule]["threshold"]
            out["fixed"][rule] = point(ids, conf, t, open_, closed)
            (out_dir / "fixed" / f"{rule}.json").write_text(json.dumps(
                {"edge": open_["edge"], "system": "gap_aware", "rule": rule, "threshold": t, "from": args.fixed,
                 "outcomes": dict(zip(ids, route(ids, conf, t, open_, closed)))}, indent=1))
    (out_dir / "curves.json").write_text(json.dumps(out, indent=1))
    (out_dir / "selected.json").write_text(json.dumps(selected, indent=1))

    def pct(x):
        return "–" if x is None else f"{100 * x:.1f}"
    ends = {"gate closed": out["curve"][0], "gate open": out["curve"][-1],
            **({f"{k} (fixed)": p for k, p in out["fixed"].items()} if args.fixed else selected)}
    print(f"{open_['edge']}: {len(ids)} requests, confidence identical in {100 * same:.0f}% "
          f"(largest difference {max(diff):.4f})")
    for name, p in ends.items():
        if p:
            print(f"  {name:15s} t={p['threshold']:.3f} sent {pct(p['escalated'])}%  >= reference "
                  f"{pct(p['at_least_as_good'])}%  hard violations {pct(p['hard_violation'])}%")


if __name__ == "__main__":
    main()
