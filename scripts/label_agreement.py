#!/usr/bin/env python
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from judge_agreement import kappa

LABELS = ["public", "abstractable", "local-only"]
SHORT = {"public": "pub", "abstractable": "abs", "local-only": "loc"}


def agree(rows: list[dict], a: str, b: str, binary: bool = False) -> str:
    pairs = [((r[a] == "local-only"), (r[b] == "local-only")) if binary else (r[a], r[b]) for r in rows]
    same = sum(x == y for x, y in pairs)
    return f"{same}/{len(pairs)} ({100 * same / len(pairs):.1f}%, kappa {kappa(pairs):.2f})"


def bootstrap(rows: list[dict], a: str, b: str, binary: bool = False, n: int = 10_000) -> str:
    by_req: dict[str, list[tuple[bool, int]]] = {}
    for r in rows:
        x, y = (r[a] == "local-only", r[b] == "local-only") if binary else (r[a], r[b])
        by_req.setdefault(r["example_id"], []).append(x == y)
    reqs = list(by_req.values())
    rng = random.Random(0)
    vals = []
    for _ in range(n):
        pick = [rng.choice(reqs) for _ in reqs]
        vals.append(sum(sum(p) for p in pick) / sum(len(p) for p in pick))
    vals.sort()
    return f"{100 * vals[int(0.025 * n)]:.1f} to {100 * vals[int(0.975 * n) - 1]:.1f}"


def main():
    sheet = Path(sys.argv[1] if len(sys.argv) > 1 else "inspection/pupa_labels_review_human.csv")
    rows = [r for r in csv.DictReader(open(sheet, encoding="utf-8-sig")) if r["human_label"] in LABELS]
    print(f"{len(rows)} spans labeled in {sheet}, {len({r['example_id'] for r in rows})} requests; "
          f"saw Claude Sonnet's label: {Counter(r['saw_sonnet_label'] for r in rows)}")
    print(f"spans marked wrong: {sum(r['span_wrong'] == 'yes' for r in rows)}, "
          f"with a comment: {sum(bool(r['comment'].strip()) for r in rows)}\n")

    print("Three labels (public / abstractable / local-only):")
    for name, col in [("final", "final_label"), ("Claude Sonnet 5.5", "sonnet_label"), ("GPT-5.5", "gpt_label")]:
        print(f"  human vs {name:18s} {agree(rows, 'human_label', col)}")
    print(f"  Claude Sonnet 5.5 vs GPT-5.5  {agree(rows, 'sonnet_label', 'gpt_label')}")
    print(f"  95% CI, human vs final (requests resampled): {bootstrap(rows, 'human_label', 'final_label')}\n")

    print("Local-only or not (the hard-violation metric):")
    for name, col in [("final", "final_label"), ("Claude Sonnet 5.5", "sonnet_label"), ("GPT-5.5", "gpt_label")]:
        print(f"  human vs {name:18s} {agree(rows, 'human_label', col, True)}")
    print(f"  95% CI, human vs final: {bootstrap(rows, 'human_label', 'final_label', True)}")
    gold_only = [r for r in rows if r["final_label"] == "local-only" and r["human_label"] != "local-only"]
    human_only = [r for r in rows if r["human_label"] == "local-only" and r["final_label"] != "local-only"]
    print(f"  local-only in gold only (would count a leak the human would not): {len(gold_only)}")
    print(f"  local-only for the human only (a leak the metric would miss): {len(human_only)}\n")

    print("Human (rows) by final label (columns):")
    conf = Counter((r["human_label"], r["final_label"]) for r in rows)
    print("          " + "".join(f"{SHORT[c]:>6s}" for c in LABELS))
    for h in LABELS:
        print(f"  {SHORT[h]:6s}  " + "".join(f"{conf[h, c]:6d}" for c in LABELS))

    for key, title in [("origin", "first marked by"), ("provenance", "final label reached by")]:
        print(f"\nBy {title}:")
        for v in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == v]
            print(f"  {v:18s} n={len(sub):3d}  three labels {agree(sub, 'human_label', 'final_label')}  "
                  f"local-only {agree(sub, 'human_label', 'final_label', True)}")

    diff = [r for r in rows if r["human_label"] != r["final_label"]]
    print(f"\nDisagreements with the final label ({len(diff)}):")
    print("  row | request | span | category | first marked by | final | human | Sonnet | GPT | reached by")
    for r in diff:
        print(f"  {r['row']} | {r['example_id']} | {r['span'][:40]} | {r['category']} | {r['origin']} | "
              f"{SHORT[r['final_label']]} | {SHORT[r['human_label']]} | {SHORT.get(r['sonnet_label'], r['sonnet_label'])} | "
              f"{SHORT.get(r['gpt_label'], r['gpt_label'])} | {r['provenance']}")


if __name__ == "__main__":
    main()
