from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from gapdeleg.privacy.matching import normalize, span_found
from gapdeleg.privacy.types import Label, Span

QUANTITY = {"AMOUNT", "MONEY"}
STOP = set("""a an the and or of to in on at for with by from as is are was were be been i me my we our you your
he she it they them his her its their this that these those each per every than then so if do does did
have has had will would can could should about into over after before up down out more less""".split())
WINDOW_WORDS = 6


@dataclass
class SpanLeak:
    text: str
    label: str
    category: str
    leaked: bool
    how: str | None
    leaked_strict: bool = False
    weak: bool = False


@dataclass
class RequestLeakage:
    request_id: str
    n_outgoing: int
    spans: list[SpanLeak] = field(default_factory=list)

    @property
    def hard_violation(self) -> bool:
        return any(s.leaked and s.label == Label.LOCAL_ONLY.value for s in self.spans)

    @property
    def hard_violation_strict(self) -> bool:
        return any(s.leaked_strict and s.label == Label.LOCAL_ONLY.value for s in self.spans)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hard_violation"] = self.hard_violation
        d["hard_violation_strict"] = self.hard_violation_strict
        return d


def is_weak(span: Span) -> bool:
    if span.category in QUANTITY:
        return True
    return span.category == "MEDICAL_CONDITION" and len(span.text.split()) <= 2


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", normalize(text))


def context_words(query: str, span: Span, k: int = 2) -> set[str]:
    if span.start < 0:
        return set()
    left = [w for w in _words(query[:span.start]) if w not in STOP and len(w) > 2][-k:]
    right = [w for w in _words(query[span.end:]) if w not in STOP and len(w) > 2][:k]
    return set(left + right)


def quantity_in_context(span: Span, query: str, union: str) -> bool:
    ctx = context_words(query, span)
    if not ctx:
        return True
    text = normalize(union)
    for m in re.finditer(rf"(?<!\w){re.escape(normalize(span.text))}(?!\w)", text):
        near = _words(text[max(0, m.start() - 60):m.end() + 60])
        if ctx & set(near[:WINDOW_WORDS * 2 + 4]):
            return True
    return False


def request_leakage(request_id: str, spans: list[Span], outgoing: list[str], threshold: float = 90,
                    min_fuzzy_len: int = 4, query: str | None = None) -> RequestLeakage:
    union = "\n".join(outgoing)
    out = RequestLeakage(request_id, len(outgoing))
    strict = {id(s): (span_found(s.text, union, threshold, min_fuzzy_len) if union else None) for s in spans}
    symptoms_found = sum(1 for s in spans if s.category == "MEDICAL_CONDITION" and strict[id(s)])
    for s in spans:
        how = strict[id(s)]
        weak = is_weak(s) and query is not None
        leaked = how is not None
        if weak and leaked:
            if s.category in QUANTITY:
                leaked = quantity_in_context(s, query, union)
            else:
                leaked = symptoms_found >= 2
            how = how if leaked else None
            how = "context" if leaked and how == "exact" else how
        out.spans.append(SpanLeak(s.text, s.label.value, s.category, leaked, how,
                                  leaked_strict=strict[id(s)] is not None, weak=is_weak(s)))
    return out


def aggregate(reqs: list[RequestLeakage]) -> dict:
    n = len(reqs)
    all_spans = [s for r in reqs for s in r.spans]
    per_label = {}
    for lab in Label:
        ss = [s for s in all_spans if s.label == lab.value]
        if ss:
            per_label[lab.value] = {"spans": len(ss), "leaked": sum(s.leaked for s in ss),
                                    "rate": sum(s.leaked for s in ss) / len(ss),
                                    "rate_strict": sum(s.leaked_strict for s in ss) / len(ss),
                                    "exact_only_rate": sum(s.how in ("exact", "context") for s in ss) / len(ss)}
    weak = [s for s in all_spans if s.weak]
    return {
        "n_requests": n,
        "requests_with_outgoing": sum(r.n_outgoing > 0 for r in reqs),
        "hard_violation_rate": sum(r.hard_violation for r in reqs) / n if n else 0.0,
        "hard_violation_rate_strict": sum(r.hard_violation_strict for r in reqs) / n if n else 0.0,
        "span_leakage_rate": sum(s.leaked for s in all_spans) / len(all_spans) if all_spans else 0.0,
        "span_leakage_rate_strict": sum(s.leaked_strict for s in all_spans) / len(all_spans) if all_spans else 0.0,
        "n_spans": len(all_spans),
        "weak_spans": {"n": len(weak), "leaked": sum(s.leaked for s in weak),
                       "leaked_strict": sum(s.leaked_strict for s in weak)},
        "abstraction_compliance": (1 - per_label["abstractable"]["rate"]) if "abstractable" in per_label else None,
        "per_label": per_label,
    }
