from __future__ import annotations

import re

from rapidfuzz import fuzz


def align(query: str, text: str, threshold: float = 88) -> tuple[int, int, str] | None:
    t = text.strip()
    if not t:
        return None
    m = re.search(rf"(?<!\w){re.escape(t)}(?!\w)", query, re.I) or re.search(re.escape(t), query, re.I)
    if m:
        return m.start(), m.end(), "exact"
    if len(t) < 4:
        return None
    a = fuzz.partial_ratio_alignment(t.lower(), query.lower())
    if a and a.score >= threshold:
        s, e = a.dest_start, a.dest_end
        while s > 0 and query[s - 1].isalnum():
            s -= 1
        while e < len(query) and query[e].isalnum():
            e += 1
        return s, e, "fuzzy"
    return None


def all_occurrences(query: str, text: str) -> list[tuple[int, int]]:
    t = text.strip()
    if not t:
        return []
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(t)}(?!\w)", query, re.I)]
