#!/usr/bin/env python
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    sheet = Path(sys.argv[1] if len(sys.argv) > 1 else "inspection/judge_check_30.csv")
    items = {str(json.loads(x)["item"]): json.loads(x) for x in open(ROOT / "data/judge_validation/check30_key.jsonl")}
    judge = {i: k["judge_verdict"] for i, k in items.items()}
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    col = next(c for c in rows[0] if c.startswith("human_verdict"))
    pairs = []
    for r in rows:
        v = r[col].strip().upper()
        if v not in ("A", "B", "TIE"):
            continue
        cand_is_a = str(items[r["item"]]["candidate_is_a"]) == "True"
        human = "tie" if v == "TIE" else "win" if (v == "A") == cand_is_a else "loss"
        pairs.append((items[r["item"]]["system"], human, judge[r["item"]]))
    print(f"{len(pairs)} of {len(rows)} rows filled in {sheet}")
    if not pairs:
        return
    for label, sub in [("all", pairs)] + [(s, [p for p in pairs if p[0] == s]) for s in sorted({p[0] for p in pairs})]:
        if not sub:
            continue
        exact = [(h, j) for _, h, j in sub]
        head = [(h in ("win", "tie"), j in ("win", "tie")) for _, h, j in sub]
        print(f"{label:12s} n={len(sub):2d}  exact {sum(a == b for a, b in exact)}/{len(sub)} (kappa {kappa(exact):.2f})  "
              f"at-least-as-good {sum(a == b for a, b in head)}/{len(sub)} (kappa {kappa(head):.2f})  "
              f"human ≥ref {sum(a for a, _ in head)}, judge ≥ref {sum(b for _, b in head)}")


if __name__ == "__main__":
    main()
