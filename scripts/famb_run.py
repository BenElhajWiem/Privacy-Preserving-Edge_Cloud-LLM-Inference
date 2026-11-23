#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))

from envfile import load_env
from gapdeleg.experiment import run

LOCAL_CLOUD = "qwen3:32b-q4_K_M"
LOCAL_CLOUD_SMALL = "qwen3:1.7b"
LOCAL_HOST = "http://localhost:11434"
CLAUDE = {"backend": "anthropic", "model": "claude-sonnet-5-5", "effort": "low"}
MAX_BUDGET = 1.0
METHOD_LABELER = "detector,edge_categorized,edge_phrases,patterns"
METHOD_GUARD = {"request_check": True, "llm_check": True, "leak_check_wording": "yes_no"}
DEV_SETS = ("privfin_dev", "privmed_dev")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def system_args(text: str) -> dict:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    out = {}
    for item in filter(None, text.split(",")):
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idea", required=True, help="folder under runs/famb/, e.g. formula")
    ap.add_argument("--variant", required=True, help="prompt variant label, e.g. terse or v0")
    ap.add_argument("--system", required=True, help="a registry name (edge_only, method_request, ...) or module:factory")
    ap.add_argument("--system-args", default="", help="keyword arguments for the system: key=value pairs separated by "
                    "commas, values read as JSON (confidence_gate=false,k=5), or a JSON object. Task lines are split "
                    "on spaces and keep quotes literally, so use the pairs there, with no spaces or quotes")
    ap.add_argument("--data", required=True, choices=DEV_SETS)
    ap.add_argument("--edge", required=True)
    ap.add_argument("--cloud", choices=["local", "claude"], default="local",
                    help="local: an Ollama model as the cloud; claude: Claude Sonnet 5.5 (needs --budget)")
    ap.add_argument("--budget", type=float, help=f"stop once this run's fresh Claude calls cost more than "
                    f"this many dollars (at most {MAX_BUDGET})")
    ap.add_argument("--local-cloud-model", help=f"Ollama model used as the cloud (default {LOCAL_CLOUD}; {LOCAL_CLOUD_SMALL} with --local)")
    ap.add_argument("--local-cloud-think", action="store_true", help="let a thinking local cloud model think (off by default)")
    ap.add_argument("--no-cloud", action="store_true", help="no cloud model at all (edge-only systems)")
    ap.add_argument("--labeler", default=METHOD_LABELER, help='comma-separated labeler parts, or "none"')
    ap.add_argument("--guard", choices=["method", "baseline"], default="method",
                    help="method: the method's guard (request check + yes/no leak check); baseline: the baselines' guard")
    ap.add_argument("--limit", type=int, help="first N requests only")
    ap.add_argument("--ids-file", help="keep only these request ids (one per line)")
    ap.add_argument("--num-ctx", type=int, default=16384, help="edge context, as the baselines")
    ap.add_argument("--local-cloud-ctx", type=int, default=8192)
    ap.add_argument("--edge-max-tokens", type=int, default=1024)
    ap.add_argument("--cloud-max-tokens", type=int, default=2048)
    ap.add_argument("--local", action="store_true", help="quick test on the local Ollama server")
    return ap


def build(args, environ=os.environ) -> tuple[dict, Path]:
    claude = args.cloud == "claude"
    if not re.fullmatch(r"[a-z0-9_-]+", args.idea):
        sys.exit("--idea: lowercase letters, digits, - and _ only")
    if claude:
        if args.no_cloud or args.local_cloud_model or args.local_cloud_think:
            sys.exit("--cloud claude: no local-cloud options and no --no-cloud")
        if args.local:
            sys.exit("--cloud claude runs on the cluster only, not with --local")
        if args.budget is None or not 0 < args.budget <= MAX_BUDGET:
            sys.exit(f"--cloud claude needs --budget, between 0 and {MAX_BUDGET} dollars")
    elif args.budget is not None:
        sys.exit("--budget is for --cloud claude only")
    if args.local:
        if not args.limit or args.limit > 5:
            sys.exit("--local is for quick tests: pass a small --limit")
        host = LOCAL_HOST
    else:
        host = environ.get("OLLAMA_HOST_URL", LOCAL_HOST)
    local_cloud = args.local_cloud_model or (LOCAL_CLOUD_SMALL if args.local else LOCAL_CLOUD)
    if claude:
        idea_dir = ROOT / "runs" / "famb" / "claude" / args.idea / ("quick" if args.limit else "")
    else:
        idea_dir = ROOT / "runs" / "famb" / "local" / args.idea / ("quick" if args.local else "")
    ids = None
    if args.ids_file:
        ids = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    if args.limit:
        from gapdeleg.data import load
        pool = [e.id for e in load("processed", name=args.data)]
        ids = [i for i in pool if ids is None or i in set(ids)][:args.limit]

    cfg = {"name": f"famb_{"claude" if claude else "local"}_{args.idea}_{slug(args.variant)}_{args.data}_{slug(args.edge)}",
           "system": args.system, "system_args": system_args(args.system_args),
           "data": {"dataset": "processed", "name": args.data, **({"ids": ids} if ids else {})},
           "edge": {"backend": "ollama", "model": args.edge, "num_ctx": args.num_ctx, "host": host},
           "labeler": [] if args.labeler == "none" else args.labeler.split(","),
           "guard": METHOD_GUARD if args.guard == "method" else {},
           "judge": None, "attacker": None,
           "gen": {"edge_max_tokens": args.edge_max_tokens, "cloud_max_tokens": args.cloud_max_tokens}}
    if claude:
        cfg["cloud"] = dict(CLAUDE)
        cfg["cloud_cache"] = "runs/cache"
        cfg["cloud_budget_usd"] = args.budget
    elif not args.no_cloud:
        cfg["cloud"] = {"backend": "ollama", "model": local_cloud, "host": host, "think": args.local_cloud_think,
                        "num_ctx": args.local_cloud_ctx}
        cfg["cloud_cache"] = "runs/famb/local-cache" + ("-quick" if args.local else "")
    return cfg, idea_dir


def main():
    args = parser().parse_args()
    cfg, idea_dir = build(args)
    claude = args.cloud == "claude"
    if claude:
        load_env()
        if not os.environ.get("GAPDELEG_ANTHROPIC_API_KEY"):
            sys.exit("GAPDELEG_ANTHROPIC_API_KEY is not set (see .env.example)")
    t = time.time()
    rd = run(cfg, run_root=idea_dir / "runs")
    s = json.loads((rd.path / "summary.json").read_text())
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
    n = len(recs)

    def mean(key):
        return sum(r["cost"][key] for r in recs) / n if n else None

    calls = mean("cloud_calls")
    fresh = calls - mean("cloud_cached_calls") if n else None
    row = {"idea": args.idea, "variant": args.variant, "data": args.data, "edge": args.edge,
           "local_cloud": None if args.no_cloud or claude else cfg["cloud"]["model"],
           "system": args.system, "system_args": cfg["system_args"],
           "n": n, "exact_match": s.get("exact_match"), "hard_violation": s["leakage"]["hard_violation_rate"],
           "local_cloud_calls": None if claude else calls, "local_cloud_fresh_calls": None if claude else fresh,
           "edge_calls": mean("edge_calls"), "guard_denials": mean("guard_denials"),
           "errors": sum(bool(r["error"]) for r in recs), "empty_answers": sum(not r["answer"] for r in recs),
           "minutes": round((time.time() - t) / 60, 1),
           "run": str(rd.path.relative_to(ROOT / "runs")), "local_quick": args.local}
    if claude:
        row.update({"cloud": cfg["cloud"]["model"], "cloud_calls": calls, "cloud_fresh_calls": fresh,
                    "cloud_spend_usd": s.get("cloud_spend_usd"), "stopped_at_budget": s.get("stopped_at_budget"),
                    "quick": bool(args.limit)})
    out = idea_dir / "rows" / f"{slug(args.variant)}__{args.data}__{slug(args.edge)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
