from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


CJK = "　-ヿ㐀-鿿가-힯豈-﫿＀-￯"


def whole_word(pattern: str, cjk: bool = False) -> str:
    w = rf"[^\W{CJK}]" if cjk else r"\w"
    return rf"(?<!{w})(?:{pattern})(?!{w})"


def exact_match(span: str, text: str) -> bool:
    span, text = normalize(span), normalize(text)
    if not span:
        return False
    return re.search(rf"(?<!\w){re.escape(span)}(?!\w)", text) is not None


def fuzzy_score(span: str, text: str) -> float:
    return fuzz.partial_ratio(normalize(span), normalize(text))


def span_found(span: str, text: str, threshold: float = 90, min_fuzzy_len: int = 4) -> str | None:
    if exact_match(span, text):
        return "exact"
    if len(normalize(span)) >= min_fuzzy_len and fuzzy_score(span, text) >= threshold:
        return "fuzzy"
    return None
