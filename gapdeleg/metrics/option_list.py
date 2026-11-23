from __future__ import annotations

import re

from gapdeleg.metrics.leakage import RequestLeakage, request_leakage

MASK = re.compile(r"\[[A-Z_]+_\d+\]")


def strip_options(message: str, options: list[str]) -> str:
    if not options:
        return message
    names = "|".join(re.escape(o) for o in sorted(options, key=len, reverse=True))
    return re.sub(rf"(?<!\w)(?:{names})(?!\w)", " ", message, flags=re.I)


def masks_an_option(message: str, options: list[str]) -> bool:
    return bool(options) and bool(MASK.search(strip_options(message, options)))


def leakage_without_options(ex, outgoing: list[str], threshold: float = 90, min_fuzzy_len: int = 4) -> RequestLeakage:
    options = ex.meta.get("options") or []
    return request_leakage(ex.id, ex.private_spans, [strip_options(m, options) for m in outgoing], threshold,
                           min_fuzzy_len, query=ex.query)
