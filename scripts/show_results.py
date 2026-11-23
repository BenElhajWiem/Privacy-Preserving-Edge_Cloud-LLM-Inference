#!/usr/bin/env python
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def pct(m):
    if not isinstance(m, dict) or m.get("value") is None or m["value"] != m["value"]:
        return "–"
    s = f"{100 * m['value']:.1f}"
    if m.get("low") is not None:
        s += f" [{100 * m['low']:.1f}, {100 * m['high']:.1f}]"
    return s


def ci_table(title, paths, metrics):
    print(f"\n=== {title}")
    print("    " + " | ".join(name for _, name in metrics))
    for path, label in paths:
        if label:
            print(f"\n--- {label}")
        for model, block in json.loads((RESULTS / path).read_text()).items():
            own = set(block["ci"]) == {"papillon"}
            print(f"\n{model}" + (" (PAPILLON with its own optimized prompts)" if own else ""))
            for system, m in block["ci"].items():
                print(f"  {system:28s} " + " | ".join(pct(m.get(key)) for key, _ in metrics))


def family_b():
    t = json.loads((RESULTS / "family_b_test/table.json").read_text())
    print("\n=== Family B test sets: exact match (%) and requests with a hard violation")
    print("    method | edge-only | redact-and-send | method minus edge-only [95% CI] | hard violations: method, edge-only, redact-and-send")
    for data, rows in t["sets"].items():
        print(f"\n{data}   cloud-only exact match: {100 * t['cloud_only'][data]['exact']:.1f}")
        for r in rows:
            d = r["vs_edge_only"]
            rs = r.get("redact_send") or {}
            print(f"  {r['edge']:30s} {100 * r['method']:5.1f} | {100 * r['edge_only']:5.1f} | "
                  f"{100 * rs.get('exact', float('nan')):5.1f} | {100 * d['diff']:+5.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}] | "
                  f"{r['hard_rule']}, {r['edge_only_hard_rule']}, {rs.get('hard_rule', '–')}")
    v = t["verdict"]["hard_rule"]
    print(f"\n  decision rule: {v['passed']} of {v['counted']} counted pairs pass (needs {v['need']})")


def main():
    quality = [("at_least_as_good", "at least as good as the reference (%)"), ("hard_violation", "hard violations (%)"),
               ("attacker_reconstruction", "attacker reconstruction (%)")]
    ci_table("Family A test set", [("family_a_test/ci.json", None)], quality)
    ci_table("Family A pilot", [("family_a_pilot/ci_qwen3/ci.json", None), ("family_a_pilot/ci_llama_gemma/ci.json", None)],
             quality)
    family_b()
    ci_table("Family B attacker", [("family_b_test/attacker/privfin/ci.json", "PrivFin"),
                                   ("family_b_test/attacker/privmed/ci.json", "PrivMed")],
             [("attacker_reconstruction", "attacker reconstruction (%)")])


if __name__ == "__main__":
    main()
