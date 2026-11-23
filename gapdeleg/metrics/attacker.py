from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from gapdeleg.privacy.matching import normalize, span_found
from gapdeleg.privacy.types import Label, Span

QUESTIONS = {
    "PERSON": "the full name of a person involved",
    "ORGANIZATION": "the name of an organization (employer, bank, school, company) involved",
    "LOCATION": "a specific place (city, venue) involved",
    "ADDRESS": "a street address",
    "EMAIL_ADDRESS": "an email address",
    "PHONE_NUMBER": "a phone number",
    "ACCOUNT_NUMBER": "an account or card number",
    "ID_NUMBER": "an ID, policy, member or case number",
    "MEDICAL_CONDITION": "a health condition or symptom of the person",
    "MONEY": "a specific amount of money",
    "AMOUNT": "a specific number or quantity",
    "DATE_TIME": "a specific date or time",
    "AGE": "the person's age",
}
DEFAULT_QUESTION = "a specific private detail about the user"

PROMPT = """You are auditing what an AI service could learn about its users. Below are all the messages
one user sent to the service{context}.

{messages}

For each item, give your single best guess of the real value, as it would appear in the user's own words.
If the messages do not reveal it, still guess the most likely value. Answer with JSON only:
{{"guesses": {{"<item id>": "<guess>", ...}}}}

Items:
{items}"""

NO_MESSAGES = "\n(no messages were sent)\n"


@dataclass
class AttributeResult:
    text: str
    category: str
    guess: str
    prior_guess: str
    reconstructed: bool
    prior_reconstructed: bool


@dataclass
class AttackResult:
    request_id: str
    attributes: list[AttributeResult] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return sum(a.reconstructed for a in self.attributes) / len(self.attributes) if self.attributes else 0.0

    @property
    def prior_rate(self) -> float:
        return sum(a.prior_reconstructed for a in self.attributes) / len(self.attributes) if self.attributes else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "rate": self.rate, "prior_rate": self.prior_rate}


def attack_targets(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in sorted([s for s in spans if s.label == Label.LOCAL_ONLY], key=lambda s: -len(s.text)):
        n = normalize(s.text)
        if n and not any(n in normalize(o.text) for o in out):
            out.append(s)
    return out


def matches(gold: str, guess: str, threshold: float = 90) -> bool:
    if not guess or not guess.strip():
        return False
    if normalize(gold) == normalize(guess):
        return True
    return len(normalize(guess)) >= len(normalize(gold)) * 0.8 and span_found(gold, guess, threshold) is not None


def match_within_category(values: list[tuple[str, str]], guesses: list[str]) -> list[bool]:
    hit = [False] * len(values)
    by_cat: dict[str, list[int]] = {}
    for i, (_, cat) in enumerate(values):
        by_cat.setdefault(cat, []).append(i)
    for idx in by_cat.values():
        fits = {v: [g for g in idx if matches(values[v][0], guesses[g])] for v in idx}
        owner: dict[int, int] = {}

        def assign(v: int, seen: set) -> bool:
            for g in fits[v]:
                if g not in seen:
                    seen.add(g)
                    if g not in owner or assign(owner[g], seen):
                        owner[g] = v
                        return True
            return False
        for v in idx:
            assign(v, set())
        for v in owner.values():
            hit[v] = True
    return hit


def rescore(attributes: list[dict]) -> list[dict]:
    values = [(a["text"], a["category"]) for a in attributes]
    hit = match_within_category(values, [a.get("guess", "") for a in attributes])
    prior = match_within_category(values, [a.get("prior_guess", "") for a in attributes])
    return [{**a, "reconstructed": h, "prior_reconstructed": p} for a, h, p in zip(attributes, hit, prior)]


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text.split("</think>")[-1], re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    g = d.get("guesses", d)
    return {str(k): str(v) for k, v in g.items()} if isinstance(g, dict) else {}


class Attacker:
    def __init__(self, cloud, guard, max_tokens: int = 2048, context: str = ""):
        self.cloud, self.guard, self.max_tokens = cloud, guard, max_tokens
        self.context = context

    def _ask(self, messages_text: str, targets: list[Span], request_id: str) -> dict:
        items = "\n".join(f'{i}: {QUESTIONS.get(s.category, DEFAULT_QUESTION)}' for i, s in enumerate(targets))
        msgs = [{"role": "user", "content": PROMPT.format(context=self.context, messages=messages_text, items=items)}]
        ticket = self.guard.evaluation_ticket(msgs, request_id=request_id)
        return _parse(self.cloud.generate(msgs, ticket=ticket, max_tokens=self.max_tokens).text)

    def attack(self, request_id: str, spans: list[Span], outgoing: list[str], prior: bool = True) -> AttackResult:
        targets = attack_targets(spans)
        res = AttackResult(request_id)
        if not targets:
            return res
        shown = "\n\n".join(f"<message>\n{m}\n</message>" for m in outgoing) if outgoing else NO_MESSAGES
        guesses = self._ask(shown, targets, request_id)
        priors = self._ask(NO_MESSAGES, targets, request_id) if prior else {}
        values = [(s.text, s.category) for s in targets]
        g = [guesses.get(str(i), "") for i in range(len(targets))]
        p = [priors.get(str(i), "") for i in range(len(targets))]
        hit, prior_hit = match_within_category(values, g), match_within_category(values, p)
        for i, s in enumerate(targets):
            res.attributes.append(AttributeResult(s.text, s.category, g[i], p[i], hit[i], prior_hit[i]))
        return res


def aggregate(results: list[AttackResult]) -> dict:
    attrs = [a for r in results for a in r.attributes]
    n = len(attrs)
    by_cat: dict[str, dict] = {}
    for a in attrs:
        c = by_cat.setdefault(a.category, {"n": 0, "reconstructed": 0, "prior_reconstructed": 0})
        c["n"] += 1
        c["reconstructed"] += a.reconstructed
        c["prior_reconstructed"] += a.prior_reconstructed
    return {"attributes": n,
            "reconstruction_rate": sum(a.reconstructed for a in attrs) / n if n else 0.0,
            "prior_rate": sum(a.prior_reconstructed for a in attrs) / n if n else 0.0,
            "requests_with_any_reconstruction": sum(any(a.reconstructed for a in r.attributes) for r in results),
            "by_category": by_cat}
