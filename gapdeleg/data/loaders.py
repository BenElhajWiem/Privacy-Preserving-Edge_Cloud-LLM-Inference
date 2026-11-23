from __future__ import annotations

import json
import os
from pathlib import Path

from gapdeleg import REPO_ROOT
from gapdeleg.privacy.types import Label, Span

from .schema import Example

EXTERNAL = Path(os.environ.get("GAPDELEG_EXTERNAL", REPO_ROOT / "external"))


def _locate(text: str, needle: str) -> tuple[int, int]:
    i = text.lower().find(needle.lower())
    return (i, i + len(needle)) if i >= 0 else (-1, -1)


def load_pupa(split: str = "tnb", limit: int | None = None, external: Path = EXTERNAL) -> list[Example]:
    import pandas as pd
    fname = {"tnb": "PUPA_TNB.csv", "new": "PUPA_New.csv"}[split.lower()]
    df = pd.read_csv(external / "PAPILLON/pupa" / fname)
    out = []
    for i, r in df.iterrows():
        q = str(r["user_query"])
        units = [u.strip() for u in str(r.get("pii_units", "") or "").split("||") if u.strip() and u.strip() != "nan"]
        spans = []
        for u in units:
            s, e = _locate(q, u)
            spans.append(Span(u, Label.LOCAL_ONLY, "GENERIC", s, e, "gold-pupa"))
        out.append(Example(id=f"pupa-{split}-{i}", query=q, reference=str(r["target_response"]),
                           private_spans=spans, family="A",
                           meta={"category": r.get("predicted_category"), "hash": r.get("conversation_hash"),
                                 "redacted_query": r.get("redacted_query")}))
        if limit and len(out) >= limit:
            break
    return out


def load_prism(domain: str | None = None, limit: int | None = None, external: Path = EXTERNAL) -> list[Example]:
    import pandas as pd
    sheets = pd.read_excel(external / "PRISM/Dataset/prism_dataset.xlsx", sheet_name=None)
    out = []
    for name, df in sheets.items():
        if domain and name.lower() != domain.lower():
            continue
        for _, r in df.iterrows():
            out.append(Example(id=f"prism-{r['id']}", query=str(r["prompt"]), family="A", meta={"domain": name}))
            if limit and len(out) >= limit:
                return out
    return out


def load_gsm8k(split: str = "test", limit: int | None = None, external: Path = EXTERNAL) -> list[Example]:
    path = external / "grade-school-math/grade_school_math/data" / f"{split}.jsonl"
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            final = r["answer"].split("####")[-1].strip().replace(",", "")
            out.append(Example(id=f"gsm8k-{split}-{i}", query=r["question"], reference=final, family="B",
                               meta={"solution": r["answer"]}))
            if limit and len(out) >= limit:
                break
    return out


def load_processed(name: str, split: str | None = None, limit: int | None = None) -> list[Example]:
    from gapdeleg.build.io import read_examples
    exs = read_examples(name)
    if split:
        exs = [e for e in exs if e.meta.get("split") == split]
    return exs[:limit] if limit else exs


LOADERS = {"pupa": load_pupa, "prism": load_prism, "gsm8k": load_gsm8k, "processed": load_processed}


def load(dataset: str, **kw) -> list[Example]:
    return LOADERS[dataset](**kw)
