#!/usr/bin/env python
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))
from envfile import load_env

PAPER = {
    "llama_32_1b": {"quality": 58.0, "leakage": 39.3},
    "llama_32_3b": {"quality": 60.9, "leakage": 24.9},
    "llama_31_8b": {"quality": 85.5, "leakage": 7.5},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--prompts", default="llama_32_3b")
    ap.add_argument("--remote", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int, help="random subset of PUPA-TNB of this size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--rescore", help="existing run dir: only re-judge its outputs")
    args = ap.parse_args()
    load_env()
    from gapdeleg.data.loaders import load_pupa
    from gapdeleg.experiment import run
    from gapdeleg.metrics.papillon_judge import papillon_scores

    ids = None
    if args.sample:
        import random
        all_ids = [e.id for e in load_pupa("tnb")]
        ids = sorted(random.Random(args.seed).sample(all_ids, args.sample), key=lambda i: int(i.rsplit("-", 1)[1]))

    if args.rescore:
        from gapdeleg.pipeline.runlog import RunDir
        rd = RunDir(Path(args.rescore).parent, run_id=Path(args.rescore).name)
        return score_and_summarize(rd, args, load_pupa, papillon_scores)
    cfg = {"name": f"papillon_repro_{args.prompts}", "system": "papillon",
           "system_args": {"prompts": args.prompts},
           "data": {"dataset": "pupa", "split": "tnb", **({"limit": args.limit} if args.limit else {}),
                    **({"ids": ids} if ids else {})},
           "edge": {"backend": "ollama", "model": args.model, "num_ctx": args.num_ctx},
           "cloud": {"backend": "openai", "model": args.remote},
           "gen": {"edge_max_tokens": 4000, "cloud_max_tokens": 4000}}
    rd = run(cfg)
    score_and_summarize(rd, args, load_pupa, papillon_scores)


def is_format_failure(prompt: str) -> bool:
    return any(m in prompt for m in ("[[ ##", "{createdPrompt}", "{reasoning}", "createdPrompt ##"))


def score_and_summarize(rd, args, load_pupa, papillon_scores):
    exs = {e.id: e for e in load_pupa("tnb")}
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]

    def score(r):
        ex = exs[r["id"]]
        prompt = r["notes"].get("created_prompt", "")
        pii = ex.meta.get("pii_str") or "||".join(s.text for s in ex.private_spans)
        if not prompt or not r["answer"] or not pii:
            return {"id": r["id"], "skipped": True}
        q, leak = papillon_scores(ex.query, ex.reference, r["answer"], prompt, pii, model=args.judge)
        return {"id": r["id"], "quality": q, "leakage": leak, "format_failure": is_format_failure(prompt)}

    with ThreadPoolExecutor(8) as pool:
        scores = list(pool.map(score, recs))
    tag = "" if args.judge == "gpt-4o-mini" else f"_{args.judge}"
    with open(rd.path / f"papillon_scores{tag}.jsonl", "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")
    ok = [s for s in scores if not s.get("skipped")]

    def ci(key, rows=None):
        import random
        rng = random.Random(1)
        vals = [s[key] for s in (rows if rows is not None else ok)]
        means = sorted(100 * sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
        return [round(means[50], 1), round(means[1949], 1)]
    summary = {"run_id": rd.id, "model": args.model, "prompts": args.prompts, "remote": args.remote,
               "n": len(recs), "scored": len(ok), "skipped": len(recs) - len(ok),
               "quality": 100 * sum(s["quality"] for s in ok) / len(ok),
               "leakage": 100 * sum(s["leakage"] for s in ok) / len(ok),
               "quality_ci95": ci("quality"), "leakage_ci95": ci("leakage"), "judge": args.judge,
               "format_failures": sum(s.get("format_failure", False) for s in ok),
               "excluding_failures": {"quality": 100 * sum(s["quality"] for s in ok if not s["format_failure"])
                                      / max(1, sum(not s["format_failure"] for s in ok)),
                                      "leakage": 100 * sum(s["leakage"] for s in ok if not s["format_failure"])
                                      / max(1, sum(not s["format_failure"] for s in ok))},
               "sample": {"size": args.sample, "seed": args.seed} if args.sample else None,
               "paper": PAPER.get(args.prompts)}
    (rd.path / f"papillon_summary{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
