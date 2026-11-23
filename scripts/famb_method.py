#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import famb_run

from envfile import load_env
from gapdeleg.experiment import run

METHOD = {"privfin": ("gapdeleg.formula.formula:make", "prompt=py_ex,mask=clean", 1.70),
          "privmed": ("method_request", "confidence_gate=false", 0.05)}
TEST_SETS = ("privfin_test", "privmed_test")
CHECK_CAP = 0.02
MAX_RERUN_CAP = 0.30
MAX_CHECK = 5


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, choices=TEST_SETS + famb_run.DEV_SETS)
    ap.add_argument("--edge", required=True)
    ap.add_argument("--limit", type=int, help=f"dev sets only: the first N requests (at most {MAX_CHECK})")
    ap.add_argument("--cap", type=float, help=f"rerun of a run stopped at its cap: its cap, at most ${MAX_RERUN_CAP}")
    return ap


def build(args, environ=os.environ) -> tuple[dict, Path]:
    test = args.data in TEST_SETS
    if test and args.limit:
        sys.exit("a test set runs whole: no --limit")
    if not test and not (args.limit and 0 < args.limit <= MAX_CHECK):
        sys.exit(f"a dev set is only a check: pass --limit {MAX_CHECK} or less")
    if args.cap is not None and not (test and 0 < args.cap <= MAX_RERUN_CAP):
        sys.exit(f"--cap is for a test set's rerun: between 0 and {MAX_RERUN_CAP} dollars")
    family = args.data.split("_")[0]
    system, system_args, cap = METHOD[family]
    dev = ["--cloud", "claude", "--budget", str(famb_run.MAX_BUDGET), "--idea", "method", "--variant", "test",
           "--system", system, "--system-args", system_args, "--data", f"{family}_dev", "--edge", args.edge]
    if args.limit:
        dev += ["--limit", str(args.limit)]
    cfg, _ = famb_run.build(famb_run.parser().parse_args(dev), environ=environ)
    cfg["name"] = f"famb_method_{args.data}_{famb_run.slug(args.edge)}"
    cfg["data"]["name"] = args.data
    cfg["cloud_budget_usd"] = args.cap or (cap if test else CHECK_CAP)
    return cfg, ROOT / "runs" / "famb-test" / "method" / ("" if test else "quick")


def main():
    args = parser().parse_args()
    cfg, folder = build(args)
    load_env()
    if not os.environ.get("GAPDELEG_ANTHROPIC_API_KEY"):
        sys.exit("GAPDELEG_ANTHROPIC_API_KEY is not set (see .env.example)")
    t = time.time()
    rd = run(cfg, run_root=folder / "runs")
    s = json.loads((rd.path / "summary.json").read_text())
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
    calls = sum(r["cost"]["cloud_calls"] for r in recs)
    row = {"data": args.data, "edge": args.edge, "system": cfg["system"], "system_args": cfg["system_args"],
           "budget_usd": cfg["cloud_budget_usd"], "n": len(recs), "exact_match": s.get("exact_match"),
           "hard_violation": s["leakage"]["hard_violation_rate"], "cloud": cfg["cloud"]["model"], "cloud_calls": calls,
           "cloud_fresh_calls": calls - sum(r["cost"]["cloud_cached_calls"] for r in recs),
           "cloud_spend_usd": s.get("cloud_spend_usd"), "stopped_at_budget": s.get("stopped_at_budget"),
           "errors": sum(bool(r["error"]) for r in recs), "empty_answers": sum(not r["answer"] for r in recs),
           "minutes": round((time.time() - t) / 60, 1), "run": str(rd.path.relative_to(ROOT / "runs")),
           "check_on_dev": bool(args.limit)}
    out = folder / "rows" / f"{args.data}__{famb_run.slug(args.edge)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
