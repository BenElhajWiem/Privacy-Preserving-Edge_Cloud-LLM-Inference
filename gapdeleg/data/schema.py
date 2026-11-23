from __future__ import annotations

from dataclasses import dataclass, field

from gapdeleg.privacy.types import Span


@dataclass
class Example:
    id: str
    query: str
    reference: str | None = None
    private_spans: list[Span] = field(default_factory=list)
    family: str = "A"
    meta: dict = field(default_factory=dict)
