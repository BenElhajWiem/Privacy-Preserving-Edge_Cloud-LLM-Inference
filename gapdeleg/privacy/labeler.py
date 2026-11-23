from __future__ import annotations

import json
import re
from typing import Protocol

from .detectors import Detector, code_values
from .policy import Policy
from .types import Label, Span, dedupe_spans


class Labeler(Protocol):
    def label(self, text: str) -> list[Span]: ...


class DetectorLabeler:
    def __init__(self, detector: Detector | None = None):
        self.detector = detector or Detector()

    def label(self, text: str) -> list[Span]:
        return dedupe_spans(self.detector.detect(text))


PROMPT = """List every piece of personal or private information in the text below: names of people,
companies and institutions, places, addresses, contact details, account or ID numbers, private amounts,
health details. Copy each one exactly as it appears. Answer with a JSON list of strings and nothing else.

Text:
{text}"""


class PromptedEdgeLabeler:
    def __init__(self, edge, max_tokens: int = 512):
        self.edge, self.max_tokens = edge, max_tokens
        self._generations = []

    def label(self, text: str) -> list[Span]:
        gen = self.edge.generate([{"role": "user", "content": PROMPT.format(text=text)}],
                                 max_tokens=self.max_tokens, temperature=0.0, logprobs=False)
        self._generations.append(gen)
        items = parse_json_list(gen.text)
        low = text.lower()
        return dedupe_spans([Span(s, Label.LOCAL_ONLY, "GENERIC", source="edge-labeler")
                             for s in items if len(s.strip()) > 1 and s.strip().lower() in low])

    def drain_generations(self) -> list:
        out, self._generations = self._generations, []
        return out


def parse_json_list(text: str) -> list[str]:
    text = text.split("</think>")[-1]
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return re.findall(r'"([^"]+)"', m.group(0))
    out: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (str, int, float)) and not isinstance(x, bool):
            out.append(str(x))
    walk(data)
    return out


CATEGORIES = ("PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "ID_NUMBER",
              "ACCOUNT_NUMBER", "URL", "DATE_TIME", "AGE", "MONEY", "MEDICAL_CONDITION", "PERSONAL_FACT")

CATEGORIZED_PROMPT = """Find every piece of private information in the text below. Private information is:
- names of people, including first names, nicknames and user names;
- companies, schools and other institutions; places and addresses; email addresses and phone numbers;
- account, ID, order and reference numbers, codes, keys and passwords;
- dates, ages and amounts of money that belong to the writer or to another real person;
- personal facts about the writer or someone they know: health, body, sexuality, feelings, relationships,
  money troubles, legal or immigration status.
Copy each one exactly as it appears in the text, and give it one category from this list:
{categories}.
Answer with a JSON list of objects like {{"text": "...", "category": "PERSON"}} and nothing else.

Text:
{text}"""


class CategorizedEdgeLabeler:

    def __init__(self, edge, policy: Policy | None = None, max_tokens: int = 768):
        self.edge, self.policy, self.max_tokens = edge, policy or Policy.load(), max_tokens
        self._generations = []

    def label(self, text: str) -> list[Span]:
        gen = self.edge.generate([{"role": "user", "content": CATEGORIZED_PROMPT.format(
            categories=", ".join(CATEGORIES), text=text)}], max_tokens=self.max_tokens, temperature=0.0,
            logprobs=False)
        self._generations.append(gen)
        low = text.lower()
        spans = []
        for value, category in parse_json_items(gen.text):
            if len(value.strip()) <= 1 or value.strip().lower() not in low:
                continue
            label = self.policy.label_for(category) if category in CATEGORIES else Label.LOCAL_ONLY
            spans.append(Span(value.strip(), label, category if category in CATEGORIES else "PRIVATE",
                              source="edge-labeler-categorized"))
        return dedupe_spans(spans)

    def drain_generations(self) -> list:
        out, self._generations = self._generations, []
        return out


def parse_json_items(text: str) -> list[tuple[str, str]]:
    text = text.split("</think>")[-1]
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [(s, "") for s in re.findall(r'"([^"]+)"', m.group(0))
                if s not in ("text", "category") and s.upper() not in CATEGORIES]

    def norm(c) -> str:
        return re.sub(r"[^A-Z]+", "_", str(c).upper()).strip("_")

    out: list[tuple[str, str]] = []

    def walk(x, category=""):
        if isinstance(x, dict):
            if "text" in x and not isinstance(x["text"], (dict, list)):
                out.append((str(x["text"]), norm(x.get("category", category))))
            else:
                for k, v in x.items():
                    walk(v, norm(k))
        elif isinstance(x, list):
            for v in x:
                walk(v, category)
        elif isinstance(x, (str, int, float)) and not isinstance(x, bool):
            out.append((str(x), category))
    walk(data)
    return out


PHRASE_PROMPT = """Below is a request that someone wants to send to an online assistant. Find the phrases
that reveal something personal and sensitive about the writer or about someone they know: health, body or
weight, sexuality or sex life, feelings, relationships, money troubles or debts, accidents, legal, police or
immigration matters (visas, embassy, court). List them even when the request asks for help with them. Copy
each phrase exactly as it appears in the request: only the words that reveal the personal fact, usually 2 to
12 words. Do not list names, places, dates or numbers on their own.

Example request: My husband left me last month and I can't pay the rent. I also want to lose weight. Write a
short letter to my landlord.
Example answer: ["My husband left me last month", "I can't pay the rent", "I want to lose weight"]

Answer with a JSON list of strings and nothing else; [] if there are none.

Request:
{text}"""


def locate(phrase: str, text: str, threshold: float = 90, min_fuzzy_len: int = 8) -> str | None:
    phrase = phrase.strip()
    i = text.lower().find(phrase.lower())
    if i >= 0:
        return text[i:i + len(phrase)]
    if len(phrase) < min_fuzzy_len:
        return None
    from rapidfuzz import fuzz
    a = fuzz.partial_ratio_alignment(phrase.lower(), text.lower())
    if a.score < threshold or a.src_end - a.src_start < len(phrase):
        return None
    start, end = a.dest_start, a.dest_end
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    return text[start:end].strip() or None


class PersonalPhraseLabeler:

    def __init__(self, edge, max_tokens: int = 384):
        self.edge, self.max_tokens = edge, max_tokens
        self._generations = []

    def label(self, text: str) -> list[Span]:
        gen = self.edge.generate([{"role": "user", "content": PHRASE_PROMPT.format(text=text)}],
                                 max_tokens=self.max_tokens, temperature=0.0, logprobs=False)
        self._generations.append(gen)
        spans = []
        for p in parse_json_list(gen.text):
            found = locate(p, text)
            if found and len(found) > 1:
                spans.append(Span(found, Label.LOCAL_ONLY, "PERSONAL_FACT", source="edge-labeler-phrases"))
        return dedupe_spans(spans)

    def drain_generations(self) -> list:
        out, self._generations = self._generations, []
        return out


class PatternLabeler:

    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy.load()

    def label(self, text: str) -> list[Span]:
        return dedupe_spans([Span(v, self.policy.label_for(cat), cat, source="pattern")
                             for cat, v in code_values(text)])


class UnionLabeler:

    def __init__(self, *labelers: Labeler):
        self.labelers = labelers

    def label(self, text: str) -> list[Span]:
        return dedupe_spans([s for lab in self.labelers for s in lab.label(text)])

    def drain_generations(self) -> list:
        return [g for lab in self.labelers for g in getattr(lab, "drain_generations", list)()]

