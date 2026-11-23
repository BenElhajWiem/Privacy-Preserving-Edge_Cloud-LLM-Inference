#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "env"))
from envfile import load_env

LOCAL_HOST = "http://localhost:11434"
CAPS = {"pupa": 0.30, "privfin": 0.10, "privmed": 0.25}
IDS = {"pupa": "configs/mac_cost_ids.txt", "privfin": "configs/mac_cost_ids_privfin.txt",
       "privmed": "configs/mac_cost_ids_privmed.txt"}


def slug(model: str) -> str:
    return model.replace(":", "-")


def read_ids(path: str) -> list[str]:
    return [x.strip() for x in (ROOT / path).read_text().splitlines() if x.strip() and not x.startswith("#")]


def pupa_config(edge: str) -> dict:
    import run_method
    from gapdeleg.build.io import request_ids
    sel = json.loads((ROOT / "runs" / "pupa-quality-dev" / "curves" / slug(edge) / "selected.json").read_text())
    threshold = sel["best_quality"]["threshold"]
    keep = set(read_ids(IDS["pupa"]))
    ids = [i for i in request_ids("pupa_tnb", None, 80, 0) if i in keep]
    return {"name": f"maccost_method_pupa_{slug(edge)}", "system": "gap_aware",
            "system_args": {"threshold": threshold, "gap_loop": False},
            "data": {"dataset": "processed", "name": "pupa_tnb", "ids": ids},
            "edge": {"backend": "ollama", "model": edge, "num_ctx": 16384, "host": LOCAL_HOST},
            "cloud": {"backend": "anthropic", "model": "claude-sonnet-5-5", "effort": "low"},
            "cloud_cache": "runs/cache", "cloud_budget_usd": CAPS["pupa"],
            "labeler": list(run_method.LABELER), "guard": dict(run_method.GUARD),
            "judge": None, "attacker": None,
            "gen": {"edge_max_tokens": 1024, "cloud_max_tokens": 2048}}


def famb_config(family: str, edge: str) -> dict:
    import famb_method
    import famb_run
    system, system_args, _ = famb_method.METHOD[family]
    dev = ["--cloud", "claude", "--budget", str(CAPS[family]), "--idea", "method", "--variant", "test",
           "--system", system, "--system-args", system_args, "--data", f"{family}_dev", "--edge", edge]
    cfg, _ = famb_run.build(famb_run.parser().parse_args(dev), environ={"OLLAMA_HOST_URL": LOCAL_HOST})
    cfg["name"] = f"maccost_method_{family}_{slug(edge)}"
    cfg["data"]["ids"] = read_ids(IDS[family])
    cfg["edge"]["host"] = LOCAL_HOST
    cfg["cloud_budget_usd"] = CAPS[family]
    return cfg


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=list(CAPS))
    ap.add_argument("--edge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the config and stop: no model calls")
    ap.add_argument("--limit", type=int, help="first N requests only (a check before the real runs)")
    args = ap.parse_args()

    cfg = pupa_config(args.edge) if args.set == "pupa" else famb_config(args.set, args.edge)
    if args.limit:
        cfg["data"]["ids"] = cfg["data"]["ids"][:args.limit]
        cfg["name"] += "_check"
    if args.dry_run:
        print(json.dumps(cfg | {"data": cfg["data"] | {"ids": f"{len(cfg['data']['ids'])} ids"}}, indent=1))
        return
    load_env()
    from gapdeleg.experiment import run
    from run_baselines import OllamaMemory

    row = {"system": f"method_{args.set}", "edge": args.edge, "data": cfg["data"]["name"],
           "system_args": cfg["system_args"]}
    if args.set == "pupa":
        row.update(threshold_rule="best_quality", threshold=cfg["system_args"]["threshold"])
    t = time.time()
    sampler = OllamaMemory(LOCAL_HOST)
    sampler.start()
    try:
        rd = run(cfg)
    finally:
        memory = sampler.stop()
    s_path = rd.path / "summary.json"
    s = json.loads(s_path.read_text())
    s["memory"] = memory
    s_path.write_text(json.dumps(s, indent=1))
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
    calls = sum(r["cost"]["cloud_calls"] for r in recs)
    row.update({"run": rd.id, "n": len(recs), "errors": sum(bool(r["error"]) for r in recs),
                "empty_answers": sum(not r["answer"] for r in recs), "memory": memory,
                "cloud_calls": calls / len(recs) if recs else 0.0,
                "cloud_fresh_calls": calls - sum(r["cost"]["cloud_cached_calls"] for r in recs),
                "cloud_spend_usd": s.get("cloud_spend_usd"), "stopped_at_budget": s.get("stopped_at_budget"),
                "edge_s": s["cost"]["edge_latency_s"]["mean"], "wall_s": s["cost"]["wall_s"]["mean"],
                "minutes": round((time.time() - t) / 60, 1)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps({"ids": cfg["data"]["ids"], "data": cfg["data"]["name"],
                                                "edge": args.edge, "cloud": cfg["cloud"]["model"], "rows": [row]},
                                               indent=1))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
