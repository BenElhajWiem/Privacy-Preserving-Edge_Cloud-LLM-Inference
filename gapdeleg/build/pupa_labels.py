from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from gapdeleg.build.align import align
from gapdeleg.build.batch import json_params, parse_json, run_batch
from gapdeleg.build.io import PROCESSED, write_examples
from gapdeleg.data.loaders import load_pupa
from gapdeleg.data.schema import Example
from gapdeleg.privacy.matching import normalize
from gapdeleg.privacy.types import Label, Span

MODEL = "claude-sonnet-5-5"
POLICY = (Path(__file__).resolve().parent.parent / "privacy/labeling_policy.txt").read_text()
SPLIT_SEED = 13
CATEGORIES = ["PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "ACCOUNT_NUMBER",
              "ID_NUMBER", "URL", "DATE_TIME", "AGE", "MONEY", "MEDICAL_CONDITION", "NRP", "OCCUPATION", "OTHER"]
LABELS = [lab.value for lab in Label]

ITEM = {"type": "object", "additionalProperties": False, "required": ["text", "label", "category", "generalized"],
        "properties": {"text": {"type": "string"}, "label": {"type": "string", "enum": LABELS},
                       "category": {"type": "string", "enum": CATEGORIES},
                       "generalized": {"type": "string", "description": "generalized form if abstractable, else empty"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["units", "missed"],
          "properties": {"units": {"type": "array", "items": ITEM},
                         "missed": {"type": "array", "items": ITEM,
                                    "description": "private items in the request that are not in the PII unit list"}}}

PROMPT = """Label the private information in this request under the policy.

<request>
{query}
</request>

PII units found by a previous annotator (lowercased, possibly misspelled):
{units}

Return every PII unit above, in the same order, with "text" copied exactly as written in the list.
Then list under "missed" any other private items in the request that the list does not cover, copying
their text exactly as it appears in the request. Leave "missed" empty if there are none."""


def build_requests(examples: list[Example]) -> dict[str, dict]:
    reqs = {}
    for ex in examples:
        units = "\n".join(f"- {s.text}" for s in ex.private_spans) or "(none)"
        reqs[ex.id] = json_params(MODEL, PROMPT.format(query=ex.query, units=units), SCHEMA, system=POLICY,
                                  max_tokens=4096, effort="low")
    return reqs


def apply_labels(ex: Example, result: dict | None) -> Example:
    parsed = parse_json(result)
    data = parsed or {"units": [], "missed": []}
    by_text = {normalize(u["text"]): u for u in data.get("units", [])}
    spans, unaligned, unlabeled = [], [], []

    def add(text, label, category, source, generalized=""):
        hit = align(ex.query, text)
        if hit is None:
            unaligned.append({"text": text, "label": label, "category": category, "source": source})
            spans.append(Span(text, Label(label), category, -1, -1, source))
            return
        s, e, how = hit
        spans.append(Span(ex.query[s:e], Label(label), category, s, e, f"{source}:{how}"))
        if generalized:
            ex.meta.setdefault("generalized", {})[ex.query[s:e]] = generalized

    for unit in ex.private_spans:
        u = by_text.get(normalize(unit.text))
        if u is None:
            unlabeled.append(unit.text)
            add(unit.text, Label.LOCAL_ONLY.value, "OTHER", "pupa-unit-unlabeled")
        else:
            add(unit.text, u["label"], u["category"], "pupa-unit", u.get("generalized", ""))
    for m in data.get("missed", []):
        add(m["text"], m["label"], m["category"], "relabel-added", m.get("generalized", ""))

    meta = dict(ex.meta, pupa_units=[s.text for s in ex.private_spans], unaligned=unaligned,
                unlabeled_units=unlabeled, label_model=MODEL,
                label_status="ok" if parsed is not None else
                f"failed ({(result or {}).get('stop_reason') or (result or {}).get('error', 'no result')}): all local-only")
    return Example(ex.id, ex.query, ex.reference, spans, ex.family, meta)


def freeze_splits(new: list[Example]) -> dict:
    ids = [ex.id for ex in new]
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)
    return {"seed": SPLIT_SEED, "pupa_tnb": {"test": [f"pupa-tnb-{i}" for i in range(237)]},
            "pupa_new": {"dev": sorted(ids[:150], key=_num), "optimization": sorted(ids[150:300], key=_num),
                         "lora_train": sorted(ids[300:], key=_num)}}


def _num(i: str) -> int:
    return int(i.rsplit("-", 1)[1])


def label_distribution(examples: list[Example]) -> dict:
    c = Counter((s.label.value, s.source.split(":")[0]) for ex in examples for s in ex.private_spans)
    by_label = Counter(s.label.value for ex in examples for s in ex.private_spans)
    by_cat = Counter(s.category for ex in examples for s in ex.private_spans)
    return {"spans": sum(by_label.values()), "by_label": dict(by_label), "by_category": dict(by_cat.most_common()),
            "by_label_and_source": {f"{k[0]} / {k[1]}": v for k, v in sorted(c.items())},
            "unaligned": sum(len(ex.meta["unaligned"]) for ex in examples),
            "unlabeled_units": sum(len(ex.meta["unlabeled_units"]) for ex in examples)}


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "env"))
    from envfile import load_env
    load_env()
    stats = {}
    splits = None
    for split in ("tnb", "new"):
        exs = load_pupa(split)
        results = run_batch(f"pupa-labels-{split}", build_requests(exs))
        labeled = [apply_labels(ex, results.get(ex.id)) for ex in exs]
        if split == "new":
            splits = freeze_splits(labeled)
            where = {i: k for k, ids in splits["pupa_new"].items() for i in ids}
            for ex in labeled:
                ex.meta["split"] = where[ex.id]
        else:
            for ex in labeled:
                ex.meta["split"] = "test"
        write_examples(f"pupa_{split}", labeled)
        stats[f"pupa_{split}"] = {"examples": len(labeled),
                                  "failed_requests": sum(ex.meta["label_status"] != "ok" for ex in labeled),
                                  **label_distribution(labeled)}
    (PROCESSED / "splits.json").write_text(json.dumps(splits, indent=1))
    (PROCESSED / "pupa_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
