from __future__ import annotations

import re
from dataclasses import dataclass, field

from gapdeleg.privacy.matching import whole_word
from gapdeleg.privacy.types import Label, Span

PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z_]*?)_(\d+)\]")


@dataclass
class Template:
    masked: str
    mapping: dict[str, str] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    cjk_words: bool = False

    def restore(self, text: str) -> str:
        return restore(text, self.mapping)

    def hide(self, text: str) -> str:
        by_lower = {original.lower(): ph for ph, original in self.mapping.items()}
        if not by_lower:
            return text
        alts = "|".join(re.escape(o) for o in sorted(by_lower, key=len, reverse=True))
        return re.sub(whole_word(alts, self.cjk_words), lambda m: by_lower.get(m.group(0).lower(), m.group(0)),
                      text, flags=re.I)

    def extend(self, values: list[str], category: str = "PRIVATE") -> "Template":
        new = [Span(v, Label.LOCAL_ONLY, category) for v in values if v.strip()]
        t = mask(self.masked, new, mapping=self.mapping, cjk_words=self.cjk_words)
        return Template(t.masked, t.mapping, self.spans + t.spans, self.cjk_words)


def _placeholder_type(span: Span) -> str:
    cat = re.sub(r"[^A-Z_]", "_", span.category.upper()).strip("_")
    return "PRIVATE" if cat in ("", "GENERIC") else cat


def mask(text: str, spans: list[Span], labels=(Label.LOCAL_ONLY, Label.ABSTRACTABLE),
         mapping: dict[str, str] | None = None, cjk_words: bool = False) -> Template:
    chosen = [s for s in spans if s.label in labels and s.text.strip()]
    occ: list[tuple[int, int, Span]] = []
    for s in chosen:
        if s.start >= 0 and text[s.start:s.end] == s.text:
            occ.append((s.start, s.end, s))
        for m in re.finditer(whole_word(re.escape(s.text.strip()), cjk_words), text, re.I):
            occ.append((m.start(), m.end(), s))
    occ.sort(key=lambda o: (-(o[1] - o[0]), o[0]))
    taken: list[tuple[int, int, Span]] = []
    for o in occ:
        if all(o[1] <= t[0] or o[0] >= t[1] for t in taken):
            taken.append(o)
    taken.sort(key=lambda o: o[0])

    counters: dict[str, int] = {}
    by_text: dict[str, str] = {}
    mapping = dict(mapping or {})
    for ph, original in mapping.items():
        by_text[original.lower()] = ph
        if m := PLACEHOLDER_RE.fullmatch(ph):
            counters[m.group(1)] = max(counters.get(m.group(1), 0), int(m.group(2)))
    out, pos = [], 0
    for start, end, span in taken:
        original = text[start:end]
        key = original.lower()
        if key not in by_text:
            typ = _placeholder_type(span)
            counters[typ] = counters.get(typ, 0) + 1
            ph = f"[{typ}_{counters[typ]}]"
            by_text[key] = ph
            mapping[ph] = original
        out.append(text[pos:start])
        out.append(by_text[key])
        pos = end
    out.append(text[pos:])
    used = {t[2] for t in taken}
    return Template("".join(out), mapping, [s for s in chosen if s in used], cjk_words)


def restore(text: str, mapping: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        return mapping.get(m.group(0), m.group(0))
    text = PLACEHOLDER_RE.sub(sub, text)
    for ph, original in mapping.items():
        bare = ph[1:-1]
        text = re.sub(rf"(?<![\w\[]){re.escape(bare)}(?![\w\]])", lambda _m, o=original: o, text)
    return text


def leftover_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)
