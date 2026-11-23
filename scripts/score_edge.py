#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.backends import make_edge
from gapdeleg.baselines import VERIFY_PROMPT, verdict_correct
from gapdeleg.build.io import read_examples, request_ids
from gapdeleg.confidence import MeanLogprob
from gapdeleg.pipeline.context import strip_think, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", required=True)
    ap.add_argument("--ids-from", help="a baseline output dir whose table.json lists the request ids to score")
    ap.add_argument("--data", default="pupa_tnb")
    ap.add_argument("--split")
    ap.add_argument("--sample", type=int, default=80, help="0 keeps every request")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="AutoMix self-verifications per request")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.ids_from:
        table = json.loads((Path(args.ids_from) / "table.json").read_text())
        data, ids = table.get("data", "pupa_tnb"), table["ids"]
    else:
        data, ids = args.data, request_ids(args.data, args.split, args.sample, args.seed)
    examples = {e.id: e for e in read_examples(data)}
    edge = make_edge({"backend": "ollama", "model": args.edge, "num_ctx": 16384})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"edge": edge.describe(), "k": args.k, "temperature": args.temperature,
                                               "data": data, "ids": ids}, indent=1))
    confidence = MeanLogprob()
    t0 = time.time()
    drafts = {}
    for n, i in enumerate(ids, 1):
        q = examples[i].query
        try:
            g = edge.generate(user(q), max_tokens=1024, logprobs=True)
        except Exception as e:
            drafts[i] = {"answer": "", "cascade_score": None, "error": f"{type(e).__name__}: {e}"}
            print(f"answer {n}/{len(ids)} {i} failed: {drafts[i]['error']}", flush=True)
            continue
        draft = strip_think(g.text)
        try:
            cascade = confidence.score(q, g)
        except ValueError:
            cascade = 0.0
        drafts[i] = {"answer": draft, "cascade_score": cascade}
        print(f"answer {n}/{len(ids)} {i} cascade {cascade:.3f}", flush=True)
    with open(out / "scores.jsonl", "w") as f:
        for n, i in enumerate(ids, 1):
            d = drafts[i]
            votes = []
            if "error" not in d:
                prompt = user(VERIFY_PROMPT.format(request=examples[i].query, response=d["answer"]))
                votes = [verdict_correct(edge.generate(prompt, max_tokens=256, temperature=args.temperature,
                                                       logprobs=False, seed=s + 1).text) for s in range(args.k)]
            automix = None if "error" in d else sum(votes) / len(votes) if votes else 0.0
            f.write(json.dumps({"id": i, **d, "automix_votes": votes, "automix_score": automix}) + "\n")
            f.flush()
            print(f"self-checks {n}/{len(ids)} {i} automix {sum(votes)}/{len(votes)}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}/scores.jsonl")


if __name__ == "__main__":
    main()
