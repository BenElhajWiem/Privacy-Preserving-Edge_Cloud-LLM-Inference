#!/usr/bin/env python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.metrics.quality import quality_summary


def main():
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p.parent for p in (ROOT / "runs").glob("*/summary.json"))
    new = {}
    for run in runs:
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        if not summary.get("quality") or not (run / "results.jsonl").exists():
            continue
        recs = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]
        q = quality_summary(recs)
        if not q:
            continue
        before = summary["quality"].get("at_least_as_good")
        if "judged_by" in summary["quality"]:
            q["judged_by"] = summary["quality"]["judged_by"]
        summary["quality"] = q
        summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        new[run.name] = q
        if before is None or abs(before - q["at_least_as_good"]) > 1e-12:
            print(f"{run.name:70s} {100 * (before or 0):5.1f}% -> {100 * q['at_least_as_good']:5.1f}% "
                  f"({q['empty_answers_as_loss']} empty answers now losses)")
    changed_tables = 0
    for table in (ROOT / "runs").glob("**/table.json"):
        t = json.loads(table.read_text())
        changed = False
        for row in t.get("rows", []):
            if row.get("run") in new:
                q = new[row["run"]]
                row.update(at_least_as_good=q["at_least_as_good"], verdicts=q["verdicts"], judge_errors=q["judge_errors"])
                changed = True
        if changed:
            table.write_text(json.dumps(t, indent=1))
            changed_tables += 1
    print(f"rescored {len(new)} runs; updated {changed_tables} tables")


if __name__ == "__main__":
    main()
