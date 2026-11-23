#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.backends import make_edge
from gapdeleg.baselines import VERIFY_PROMPT
from gapdeleg.build.io import read_examples, request_ids
from gapdeleg.pipeline.context import strip_think, user

PASSES = ["order", "order_again", "reverse", "interleaved"]


def answer(edge, q):
    try:
        return strip_think(edge.generate(user(q), max_tokens=1024, logprobs=True).text)
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", required=True)
    ap.add_argument("--data", default="pupa_tnb")
    ap.add_argument("--split")
    ap.add_argument("--sample", type=int, default=80, help="0 keeps every request")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--against", help="a stored edge-only run directory (its results.jsonl) from another job")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = request_ids(args.data, args.split, args.sample, args.seed)
    examples = {e.id: e for e in read_examples(args.data)}
    edge = make_edge({"backend": "ollama", "model": args.edge, "num_ctx": 16384})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = {p: {} for p in PASSES}
    t0 = time.time()
    for p in PASSES:
        order = ids[::-1] if p == "reverse" else ids
        for n, i in enumerate(order, 1):
            q = examples[i].query
            answers[p][i] = answer(edge, q)
            if p == "interleaved":
                edge.generate(user(VERIFY_PROMPT.format(request=q, response=answers[p][i])), max_tokens=256,
                              temperature=0.7, logprobs=False, seed=1)
            print(f"{p} {n}/{len(ids)} {i} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    with open(out / "answers.jsonl", "w") as f:
        for i in ids:
            f.write(json.dumps({"id": i, **{p: answers[p][i] for p in PASSES}}) + "\n")

    def same(a, b):
        keep = [i for i in ids if i in a and i in b and not a[i].startswith("ERROR") and not b[i].startswith("ERROR")]
        return {"identical": sum(a[i] == b[i] for i in keep), "compared": len(keep)}

    summary = {"edge": edge.describe(), "data": args.data, "ids": len(ids),
               "errors": {p: sum(a.startswith("ERROR") for a in answers[p].values()) for p in PASSES},
               "vs_order": {p: same(answers["order"], answers[p]) for p in PASSES[1:]}}
    if args.against:
        stored = {r["id"]: r["answer"] or "" for r in map(json.loads, open(Path(args.against) / "results.jsonl"))}
        summary["against"] = args.against
        summary["vs_stored_run"] = {p: same(stored, answers[p]) for p in PASSES}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for k in ("vs_order", "vs_stored_run"):
        for p, s in summary.get(k, {}).items():
            print(f"{k:14} {p:12} identical {s['identical']}/{s['compared']}")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
