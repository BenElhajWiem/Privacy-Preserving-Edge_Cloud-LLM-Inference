#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))
from envfile import load_env

from gapdeleg.backends import make_cloud
from gapdeleg.data import load
from gapdeleg.experiment import cloud_spend
from gapdeleg.metrics.attacker import Attacker
from gapdeleg.metrics.attacker import aggregate as attack_aggregate
from gapdeleg.pipeline.egress import load_outgoing
from gapdeleg.privacy.guard import EgressGuard
from gapdeleg.privacy.policy import Policy


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--cloud", default="claude-sonnet-5-5")
    ap.add_argument("--budget", type=float, required=True, help="dollars of fresh attacker calls, over all runs")
    ap.add_argument("--root", type=Path, default=ROOT, help="checkout whose runs/ and .env to use")
    ap.add_argument("--dry-run", action="store_true", help="check the tasks and make no calls")
    args = ap.parse_args()
    runs_dir = args.root.resolve() / "runs"
    load_env(args.root.resolve() / ".env")
    model = make_cloud({"backend": "anthropic", "model": args.cloud, "effort": "low"}, cache_dir=runs_dir / "cache")
    for task in map(Path, args.tasks):
        table_path = task / "table.json" if task.is_dir() else task
        table = json.loads(table_path.read_text())
        rows = table["rows"] if "rows" in table else [table]
        run = runs_dir / rows[0]["run"]
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        policy = Policy.load(cfg["policy"]) if cfg.get("policy") else Policy.load()
        attacker = Attacker(model, EgressGuard(policy, log_path=run / "guard.jsonl"))
        data = dict(cfg["data"])
        ids = data.pop("ids", None)
        examples = load(data.pop("dataset"), **data)
        if ids:
            keep = set(ids)
            examples = [ex for ex in examples if ex.id in keep]
        outgoing = load_outgoing(run / "outgoing.jsonl")
        if args.dry_run:
            print(json.dumps({"task": str(task), "run": run.name, "requests": len(examples),
                              "with_messages": sum(bool(outgoing.get(ex.id)) for ex in examples),
                              "has_attacker": (run / "attacker.json").exists()}), flush=True)
            continue
        attacks = []
        for ex in examples:
            if cloud_spend(model) > args.budget:
                print(f"budget of ${args.budget} reached before {run.name} / {ex.id}: nothing written for this run",
                      flush=True)
                sys.exit(1)
            attacks.append(attacker.attack(ex.id, ex.private_spans, outgoing.get(ex.id, [])))
        (run / "attacker.json").write_text(json.dumps([a.to_dict() for a in attacks], indent=1, ensure_ascii=False))
        summary = json.loads((run / "summary.json").read_text())
        summary["attacker"] = attack_aggregate(attacks)
        summary["attacker_scoring"] = "within-category"
        summary["attacker_added_by"] = "scripts/attack_stored.py"
        (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        for row in rows:
            row["attacker_reconstruction"] = summary["attacker"]["reconstruction_rate"]
            row["attacker_prior"] = summary["attacker"]["prior_rate"]
        table_path.write_text(json.dumps(table, indent=1))
        print(json.dumps({"task": str(task), "run": run.name, "requests": len(attacks),
                          "reconstruction": summary["attacker"]["reconstruction_rate"],
                          "prior": summary["attacker"]["prior_rate"], "spend_so_far": round(cloud_spend(model), 4)}),
              flush=True)
    print(f"fresh attacker calls: ${cloud_spend(model):.4f}")


if __name__ == "__main__":
    main()
