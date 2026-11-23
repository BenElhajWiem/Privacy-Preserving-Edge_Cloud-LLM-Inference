from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Label(str, Enum):
    PUBLIC = "public"
    ABSTRACTABLE = "abstractable"
    LOCAL_ONLY = "local-only"


@dataclass(frozen=True)
class Span:
    text: str
    label: Label = Label.LOCAL_ONLY
    category: str = "GENERIC"
    start: int = -1
    end: int = -1
    source: str = ""
    score: float = 1.0
    provenance: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(**{**d, "label": Label(d["label"])})


def dedupe_spans(spans: list[Span]) -> list[Span]:
    order = {Label.PUBLIC: 0, Label.ABSTRACTABLE: 1, Label.LOCAL_ONLY: 2}
    best: dict[str, Span] = {}
    for s in spans:
        k = s.text.strip().lower()
        if not k:
            continue
        if k not in best or order[s.label] > order[best[k].label]:
            best[k] = s
    return list(best.values())
