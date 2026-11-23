#!/usr/bin/env python
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))
from envfile import load_env

JUDGE_MAX_TOKENS = 2048


def run_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "results.jsonl").exists():
            out.append(p)
        elif (p / "table.json").exists():
            out += [ROOT / "runs" / r["run"] for r in json.loads((p / "table.json").read_text())["rows"] if "run" in r]
    return list(dict.fromkeys(out))


def pending(run: Path, examples: dict, force: bool) -> list[tuple[dict, object]]:
    recs = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]
    todo = []
    for r in recs:
        ex = examples.get(r["id"])
        if ex and ex.family == "A" and ex.reference and r["answer"] and (force or "judge" not in r):
            todo.append((r, ex))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--batch", action="store_true", help="send uncached judge prompts as one OpenAI batch first")
    ap.add_argument("--force", action="store_true", help="re-judge answers that already have a verdict")
    args = ap.parse_args()
    load_env()

    from gapdeleg.backends import make_cloud
    from gapdeleg.backends.base import Generation, Usage
    from gapdeleg.build.io import read_examples
    from gapdeleg.build.openai_batch import run_openai_batch
    from gapdeleg.metrics.quality import JUDGE_PROMPT, PairwiseJudge, quality_summary
    from gapdeleg.privacy.detectors import Detector
    from gapdeleg.privacy.guard import EgressGuard
    from gapdeleg.privacy.policy import Policy

    runs = run_dirs(args.runs)
    judge_model = make_cloud({"backend": "openai"}, cache_dir=ROOT / "runs/cache")
    params = {"max_tokens": JUDGE_MAX_TOKENS, "temperature": 0.0, "logprobs": False}
    datasets: dict[str, dict] = {}
    work = {}
    for run in runs:
        name = yaml.safe_load((run / "config.yaml").read_text())["data"]["name"]
        if name not in datasets:
            datasets[name] = {e.id: e for e in read_examples(name)}
        work[run] = pending(run, datasets[name], args.force)
    print(f"{sum(map(len, work.values()))} answers to judge in {len(runs)} runs")

    if args.batch:
        requests, keys = {}, {}
        for todo in work.values():
            for r, ex in todo:
                for a, b in ((r["answer"], ex.reference), (ex.reference, r["answer"])):
                    msgs = [{"role": "user", "content": JUDGE_PROMPT.format(query=ex.query, a=a, b=b)}]
                    key = judge_model._cache_key(msgs, params)
                    if key in keys.values() or judge_model._cache_get(key):
                        continue
                    cid = f"j{len(keys)}"
                    keys[cid] = key
                    body = {"model": judge_model.name, "messages": msgs}
                    if judge_model.reasoning_model:
                        body.update(max_completion_tokens=JUDGE_MAX_TOKENS, reasoning_effort=judge_model.reasoning_effort)
                    else:
                        body.update(max_tokens=JUDGE_MAX_TOKENS, temperature=0.0)
                    requests[cid] = body
        print(f"{len(requests)} judge prompts not in the cache go to the batch")
        if requests:
            res = run_openai_batch(f"judge-{time.strftime('%Y%m%d-%H%M%S')}", requests)
            stored = 0
            for cid, r in res.items():
                if cid in keys and r.get("ok") and r.get("text"):
                    u = r.get("usage") or {}
                    judge_model._cache_put(keys[cid], Generation(r["text"], usage=Usage(
                        u.get("input_tokens", 0), u.get("output_tokens", 0)), model=judge_model.name))
                    stored += 1
            print(f"stored {stored} of {len(requests)} batch replies in runs/cache")

    policy = Policy.load()
    detector = Detector(policy)
    judged_runs = {}
    for run, todo in work.items():
        if not todo:
            continue
        guard = EgressGuard(policy, detector=detector, log_path=run / "guard.jsonl")
        judge = PairwiseJudge(judge_model, guard)
        by_id = {r["id"]: r for r, _ in todo}
        for r, ex in todo:
            try:
                r["judge"] = asdict(judge.compare(ex.query, r["answer"], ex.reference, request_id=ex.id))
                r.pop("judge_error", None)
            except Exception as e:
                r.pop("judge", None)
                r["judge_error"] = f"{type(e).__name__}: {e}"[:300]
        recs = [by_id.get(x["id"], x) for x in map(json.loads, (run / "results.jsonl").read_text().splitlines())]
        (run / "results.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
        summary = json.loads((run / "summary.json").read_text())
        summary["quality"] = dict(quality_summary(recs) or {"at_least_as_good": None},
                                  judged_by="judge_runs.py" + (" (batch)" if args.batch else ""))
        (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        judged_runs[run.name] = summary["quality"]
        print(f"{run.name}: at least as good {100 * summary['quality']['at_least_as_good']:.1f}% "
              f"({sum(summary['quality'].get('verdicts', {}).values())} judged, "
              f"{summary['quality'].get('empty_answers_as_loss', 0)} empty, {summary['quality'].get('judge_errors', 0)} unjudged)")
    for table in (ROOT / "runs").glob("**/table.json"):
        t = json.loads(table.read_text())
        changed = False
        for row in t.get("rows", []):
            if row.get("run") in judged_runs:
                q = judged_runs[row["run"]]
                row.update(at_least_as_good=q["at_least_as_good"], verdicts=q["verdicts"], judge_errors=q["judge_errors"])
                changed = True
        if changed:
            table.write_text(json.dumps(t, indent=1))
    live = judge_model.spent
    print(f"live judge tokens this run: {live.input_tokens} in / {live.output_tokens} out "
          f"(0 means every verdict came from the cache or the batch)")


if __name__ == "__main__":
    main()
