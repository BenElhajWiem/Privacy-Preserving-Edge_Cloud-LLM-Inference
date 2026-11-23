#!/usr/bin/env python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.metrics.attacker import AttackResult, AttributeResult, aggregate, rescore


def rescore_run(run: Path) -> dict | None:
    summary_path, attack_path = run / "summary.json", run / "attacker.json"
    if not attack_path.exists() or not summary_path.exists():
        return None
    results = []
    for r in json.loads(attack_path.read_text()):
        attrs = rescore(r["attributes"])
        results.append(AttackResult(r["request_id"], [AttributeResult(**{k: a[k] for k in (
            "text", "category", "guess", "prior_guess", "reconstructed", "prior_reconstructed")}) for a in attrs]))
    attack_path.write_text(json.dumps([a.to_dict() for a in results], indent=1, ensure_ascii=False))
    summary = json.loads(summary_path.read_text())
    before = summary.get("attacker", {}).get("reconstruction_rate")
    summary["attacker"] = aggregate(results)
    summary["attacker_scoring"] = "within-category"
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return {"before": before, "after": summary["attacker"]["reconstruction_rate"],
            "prior": summary["attacker"]["prior_rate"]}


def main():
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p.parent for p in (ROOT / "runs").glob("*/attacker.json"))
    new = {}
    for run in runs:
        r = rescore_run(run)
        if r:
            new[run.name] = r
            print(f"{run.name:60s} {100 * (r['before'] or 0):5.1f}% -> {100 * r['after']:5.1f}%")
    for table in (ROOT / "runs").glob("**/table.json"):
        t = json.loads(table.read_text())
        changed = False
        for row in t.get("rows", []):
            if row.get("run") in new:
                row["attacker_reconstruction"] = new[row["run"]]["after"]
                row["attacker_prior"] = new[row["run"]]["prior"]
                changed = True
        if changed:
            table.write_text(json.dumps(t, indent=1))
    print(f"rescored {len(new)} runs")


if __name__ == "__main__":
    main()
