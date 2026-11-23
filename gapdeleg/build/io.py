from __future__ import annotations

import json
import re
from pathlib import Path

from gapdeleg import REPO_ROOT
from gapdeleg.data.schema import Example
from gapdeleg.privacy.types import Span

PROCESSED = REPO_ROOT / "data/processed"


def example_to_dict(ex: Example) -> dict:
    return {"id": ex.id, "family": ex.family, "query": ex.query, "reference": ex.reference,
            "private_spans": [s.to_dict() for s in ex.private_spans], "meta": ex.meta}


def example_from_dict(d: dict) -> Example:
    return Example(id=d["id"], query=d["query"], reference=d.get("reference"), family=d.get("family", "A"),
                   private_spans=[Span.from_dict(s) for s in d.get("private_spans", [])], meta=d.get("meta", {}))


def write_examples(name: str, examples: list[Example]) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    p = PROCESSED / f"{name}.jsonl"
    with open(p, "w") as f:
        for ex in examples:
            f.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")
    return p


def read_examples(name: str, processed: Path = PROCESSED) -> list[Example]:
    return [example_from_dict(json.loads(x)) for x in (processed / f"{name}.jsonl").read_text().splitlines()]


def id_order(request_id: str) -> int:
    return int(re.search(r"\d+$", request_id).group())


def request_ids(name: str, split: str | None = None, sample: int = 0, seed: int = 0) -> list[str]:
    import random
    pool = [e.id for e in read_examples(name)]
    if split:
        keep = set(json.loads((PROCESSED / "splits.json").read_text())[name][split])
        pool = [i for i in pool if i in keep]
    if sample and sample < len(pool):
        pool = random.Random(seed).sample(pool, sample)
    return sorted(pool, key=id_order)
