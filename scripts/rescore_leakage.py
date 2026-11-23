#!/usr/bin/env python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.build.review_pupa import PLACEHOLDER, PRONOUN_FIXES
from gapdeleg.metrics import leakage as leak_m
from gapdeleg.metrics.attacker import AttackResult, AttributeResult
from gapdeleg.metrics.attacker import aggregate as attack_aggregate

ATTR_FIELDS = ("text", "category", "guess", "prior_guess", "reconstructed", "prior_reconstructed")
LEAK_COLUMNS = {"hard_violation": lambda s: s["leakage"]["hard_violation_rate"],
                "hard_violation_strict": lambda s: s["leakage"]["hard_violation_rate_strict"],
                "span_leak_local_only": lambda s: s["leakage"]["per_label"].get("local-only", {}).get("rate"),
                "abstraction_compliance": lambda s: s["leakage"]["abstraction_compliance"],
                "attacker_reconstruction": lambda s: s.get("attacker", {}).get("reconstruction_rate"),
                "attacker_prior": lambda s: s.get("attacker", {}).get("prior_rate")}


def artifact(request_id: str, text: str) -> bool:
    return bool(PLACEHOLDER.match(text.strip())) or (request_id, text) in PRONOUN_FIXES


def fix_run(run: Path) -> dict | None:
    recs = json.loads((run / "leakage.json").read_text())
    relabeled, reqs = 0, []
    for r in recs:
        spans = []
        for s in r["spans"]:
            if s["label"] != "public" and artifact(r["request_id"], s["text"]):
                s = dict(s, label="public")
                relabeled += 1
            spans.append(leak_m.SpanLeak(**s))
        reqs.append(leak_m.RequestLeakage(r["request_id"], r["n_outgoing"], spans))
    attacks, dropped = None, 0
    if (run / "attacker.json").exists():
        attacks = []
        for r in json.loads((run / "attacker.json").read_text()):
            keep = [a for a in r["attributes"] if not artifact(r["request_id"], a["text"])]
            dropped += len(r["attributes"]) - len(keep)
            attacks.append(AttackResult(r["request_id"], [AttributeResult(**{k: a[k] for k in ATTR_FIELDS})
                                                          for a in keep]))
    if not relabeled and not dropped:
        return None
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    out = {"spans": relabeled, "attacker_attributes": dropped, "n": len(reqs),
           "hard_before": sum(bool(r.get("hard_violation")) for r in recs),
           "hard_after": sum(q.hard_violation for q in reqs)}
    (run / "leakage.json").write_text(json.dumps([q.to_dict() for q in reqs], indent=1, ensure_ascii=False))
    summary["leakage"] = leak_m.aggregate(reqs)
    if dropped:
        (run / "attacker.json").write_text(json.dumps([a.to_dict() for a in attacks], indent=1, ensure_ascii=False))
        summary["attacker"] = attack_aggregate(attacks)
    summary["leakage_labels"] = "artifact fix 2026-11-08 (scripts/rescore_leakage.py)"
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    out["summary"] = summary
    return out


def main():
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p.parent for p in (ROOT / "runs").glob("*/leakage.json"))
    changed = {}
    for run in runs:
        try:
            r = fix_run(run)
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"SKIPPED {run.name}: {type(e).__name__}: {e}")
            continue
        if r:
            changed[run.name] = r
            print(f"{run.name:75s} {r['spans']:3d} spans, {r['attacker_attributes']:2d} attacker attributes, "
                  f"hard violations {r['hard_before']} -> {r['hard_after']} of {r['n']}")
    tables = 0
    for table in (ROOT / "runs").glob("**/table.json"):
        t = json.loads(table.read_text())
        rows = [row for row in t.get("rows", []) if row.get("run") in changed]
        for row in rows:
            s = changed[row["run"]]["summary"]
            for key, get in LEAK_COLUMNS.items():
                if key in row:
                    row[key] = get(s)
        if rows:
            table.write_text(json.dumps(t, indent=1))
            tables += 1
    print(f"changed {len(changed)} runs and {tables} tables")


if __name__ == "__main__":
    main()
