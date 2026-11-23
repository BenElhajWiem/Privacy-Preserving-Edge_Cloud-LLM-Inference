from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict

from gapdeleg import REPO_ROOT
from gapdeleg.backends import make_cloud
from gapdeleg.build.batch import cost, log_cost
from gapdeleg.data.loaders import load_pupa
from gapdeleg.experiment import run
from gapdeleg.metrics.quality import PairwiseJudge
from gapdeleg.privacy.guard import EgressGuard

SEED = 21
OUT = REPO_ROOT / "data/judge_validation"
EDGE = {"backend": "ollama", "model": "qwen3:1.7b", "think": False}
CLOUD = {"backend": "anthropic", "model": "claude-haiku-5-5", "effort": "low"}


def pick_ids() -> tuple[list[str], list[str]]:
    ids = [e.id for e in load_pupa("tnb")]
    rng = random.Random(SEED)
    rng.shuffle(ids)
    return ids[:30], ids[30:60]


def main():
    sys.path.insert(0, str(REPO_ROOT / "env"))
    from envfile import load_env
    load_env()
    edge_ids, redact_ids = pick_ids()
    base = {"data": {"dataset": "pupa", "split": "tnb"}, "edge": EDGE, "gen": {"edge_max_tokens": 1024,
                                                                                "cloud_max_tokens": 1024}}
    runs = {
        "edge_only": run({**base, "name": "judgeset_edge_only", "system": "edge_only",
                          "data": {**base["data"], "ids": edge_ids}}),
        "redact_send": run({**base, "name": "judgeset_redact_send", "system": "redact_send", "cloud": CLOUD,
                            "labeler": ["detector", "edge"], "data": {**base["data"], "ids": redact_ids}}),
    }
    exs = {e.id: e for e in load_pupa("tnb")}
    rng = random.Random(SEED + 1)
    items = []
    for system, rd in runs.items():
        for line in rd.results.read_text().splitlines():
            r = json.loads(line)
            ex = exs[r["id"]]
            candidate_is_a = rng.random() < 0.5
            items.append({"item": None, "id": ex.id, "system": system, "run_id": rd.id, "query": ex.query,
                          "candidate": r["answer"], "reference": ex.reference, "candidate_is_a": candidate_is_a})
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item"] = i
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    judge_model = make_cloud({"backend": "openai"}, cache_dir=REPO_ROOT / "runs/cache")
    judge = PairwiseJudge(judge_model, EgressGuard())
    with open(OUT / "judge.jsonl", "w") as f:
        for it in items:
            res = judge.compare(it["query"], it["candidate"], it["reference"], request_id=it["id"])
            f.write(json.dumps({"item": it["item"], "id": it["id"], "judge_model": judge_model.name,
                                "verdict": res.verdict, "order_ab": res.order_ab, "order_ba": res.order_ba}) + "\n")
    usage = asdict(judge_model.spent)
    log_cost("judge-validation", judge_model.name, usage, cost(judge_model.name, usage, batch=False), 2 * len(items),
             batch=False)
    print(f"judge set: {len(items)} items -> {OUT}; judge tokens {usage}")


if __name__ == "__main__":
    main()
