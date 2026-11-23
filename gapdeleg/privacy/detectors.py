from __future__ import annotations

import logging
import re
from functools import lru_cache

from .policy import Policy
from .types import Span

PRESIDIO_MAP = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL_ADDRESS", "PHONE_NUMBER": "PHONE_NUMBER",
                "CREDIT_CARD": "CREDIT_CARD", "IBAN_CODE": "IBAN_CODE", "US_SSN": "US_SSN",
                "IP_ADDRESS": "IP_ADDRESS", "LOCATION": "LOCATION", "DATE_TIME": "DATE_TIME",
                "NRP": "NRP", "URL": "URL", "US_BANK_NUMBER": "ACCOUNT_NUMBER",
                "US_DRIVER_LICENSE": "ID_NUMBER", "US_PASSPORT": "ID_NUMBER", "UK_NHS": "ID_NUMBER",
                "MEDICAL_LICENSE": "ID_NUMBER", "ORGANIZATION": "ORGANIZATION"}

REGEXES = {
    "ACCOUNT_NUMBER": re.compile(r"\b(?:acct|account|a/c)\.?\s*(?:no\.?|number|#)?\s*[:#]?\s*\d[\d\- ]{5,}\d\b", re.I),
    "ID_NUMBER": re.compile(r"\b[A-Z]{1,3}\d{6,10}\b"),
    "MONEY": re.compile(r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|million|billion))?)", re.I),
}

CODE_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.A),
    "ID_NUMBER": re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
        r"|\b[0-9a-fA-F]{16,}\b"
        r"|(?<![\w@.])@[A-Za-z_]\w{2,}"
        r"|\b(?=\w*[A-Za-z]\d+[A-Za-z])\w*[A-Za-z]\d\w*\b"
        r"|\b(?=\w*\d+[A-Za-z]+\d)\w*\d[A-Za-z]\w*\b"
        r"|(?<![\w.])\d(?:[\d\- ]?\d){6,}(?![\w.])", re.A),
}


def code_values(text: str) -> list[tuple[str, str]]:
    out = []
    for cat, rx in CODE_PATTERNS.items():
        for m in rx.finditer(text):
            v = m.group(0).strip()
            if m.start() > 0 and text[m.start() - 1] == "#":
                continue
            if len(v) >= 4 and not PLACEHOLDER.search(v) and not re.fullmatch(r"[A-Z_]+_\d+", v):
                out.append((cat, v))
    return out


PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


@lru_cache(maxsize=1)
def _analyzer():
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    logging.getLogger("tldextract").setLevel(logging.CRITICAL)
    from presidio_analyzer import AnalyzerEngine
    return AnalyzerEngine()


class Detector:
    def __init__(self, policy: Policy | None = None, use_presidio: bool = True, use_regex: bool = True,
                 min_score: float = 0.4):
        self.policy = policy or Policy.load()
        self.use_presidio, self.use_regex, self.min_score = use_presidio, use_regex, min_score

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        if self.use_presidio:
            for r in _analyzer().analyze(text=text, language="en"):
                if r.score < self.min_score:
                    continue
                frag = text[r.start:r.end]
                if PLACEHOLDER.fullmatch(frag.strip()) or PLACEHOLDER.search(frag):
                    continue
                cat = PRESIDIO_MAP.get(r.entity_type, r.entity_type)
                spans.append(Span(frag, self.policy.label_for(cat), cat, r.start, r.end, "presidio", r.score))
        if self.use_regex:
            for cat, rx in REGEXES.items():
                for m in rx.finditer(text):
                    spans.append(Span(m.group(0), self.policy.label_for(cat), cat, m.start(), m.end(), "regex"))
        return spans
