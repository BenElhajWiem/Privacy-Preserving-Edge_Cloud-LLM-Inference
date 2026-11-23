#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))
from envfile import load_env

LABELER = ["detector", "edge_categorized", "edge_phrases", "patterns"]
GUARD = {"request_check": True, "llm_check": True, "leak_check_wording": "yes_no"}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", required=True)
    ap.add_argument("--system", default="gap_aware", choices=["gap_aware", "method_request"])
    ap.add_argument("--gate", required=True, help='"open", "closed" or a confidence threshold')
    ap.add_argument("--no-phrases", action="store_true", help="leave the labeler's phrase pass out")
    ap.add_argument("--data", default="pupa_tnb")
    ap.add_argument("--split")
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-file", help="keep only these request ids (one per line), e.g. configs/pupa_test_ids.txt")
    ap.add_argument("--cloud", default="claude-sonnet-5-5")
    ap.add_argument("--cloud-backend", choices=["anthropic", "ollama"], default="anthropic",
                    help="ollama: use a local Ollama model (--cloud <tag>) as the cloud, no API key needed")
    ap.add_argument("--cloud-cache", default="runs/cache")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--attacker", action="store_true")
    ap.add_argument("--max-cloud-usd", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N requests (trials)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load_env()
    from gapdeleg.build.io import request_ids
    from gapdeleg.experiment import run

    ids = request_ids(args.data, args.split, args.sample, args.seed)
    if args.ids_file:
        keep = set(Path(args.ids_file).read_text().split())
        ids = [i for i in ids if i in keep]
    ids = ids[:args.limit] if args.limit else ids
    if args.gate == "open":
        flags = {"confidence_gate": False}
    elif args.gate == "closed":
        flags = {"threshold": 0.0}
    else:
        flags = {"threshold": float(args.gate)}
    label = f"{'method' if args.system == 'gap_aware' else 'mreq'}_{args.gate}{'-nophrases' if args.no_phrases else ''}_{args.edge.replace(':', '-')}"
    cfg = {"name": label, "system": args.system, "system_args": flags,
           "data": {"dataset": "processed", "name": args.data, "ids": ids},
           "edge": {"backend": "ollama", "model": args.edge, "num_ctx": 16384},
           "cloud": {"backend": "ollama", "model": args.cloud} if args.cloud_backend == "ollama" else {"backend": "anthropic", "model": args.cloud, "effort": "low"},
           "cloud_cache": args.cloud_cache, "cloud_budget_usd": args.max_cloud_usd,
           "labeler": [x for x in LABELER if not (args.no_phrases and x == "edge_phrases")], "guard": GUARD,
           "judge": None if args.no_judge else {"backend": "openai"},
           "attacker": {"backend": "anthropic", "model": args.cloud, "effort": "low"} if args.attacker else None,
           "gen": {"edge_max_tokens": 1024, "cloud_max_tokens": 2048}}
    t = time.time()
    rd = run(cfg)
    s = json.loads((rd.path / "summary.json").read_text())
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
    row = {"system": args.system, "edge": args.edge, "gate": args.gate, "phrases": not args.no_phrases,
           "run": rd.id, "n": len(recs), "errors": sum(bool(r["error"]) for r in recs),
           "empty_answers": sum(not r["answer"] for r in recs),
           "at_least_as_good": s.get("quality", {}).get("at_least_as_good"), "exact_match": s.get("exact_match"),
           "hard_violation": s["leakage"]["hard_violation_rate"],
           "cloud_calls": s["cost"]["cloud_calls"]["mean"], "cloud_spend_usd": s.get("cloud_spend_usd"),
           "stopped_at_budget": s.get("stopped_at_budget"), "minutes": round((time.time() - t) / 60, 1)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps({"ids": ids, "data": args.data, "split": args.split, "seed": args.seed,
                                                "edge": args.edge, "cloud": args.cloud, "rows": [row]}, indent=1))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
