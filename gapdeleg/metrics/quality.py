from __future__ import annotations

import re
from dataclasses import dataclass

JUDGE_PROMPT = """You are comparing two responses to the same user request.

Request:
{query}

Response A:
{a}

Response B:
{b}

Which response better fulfils the request? Answer with exactly one of: A, B, TIE."""


@dataclass
class PairwiseResult:
    verdict: str
    order_ab: str
    order_ba: str


def _parse(text: str) -> str:
    t = text.strip().upper()
    m = re.search(r"\b(TIE|A|B)\b", t)
    return m.group(1) if m else "TIE"


class PairwiseJudge:

    def __init__(self, cloud, guard, max_tokens: int = 2048):
        self.cloud, self.guard, self.max_tokens = cloud, guard, max_tokens

    def _ask(self, query, a, b, request_id) -> str:
        msgs = [{"role": "user", "content": JUDGE_PROMPT.format(query=query, a=a, b=b)}]
        ticket = self.guard.evaluation_ticket(msgs, request_id=request_id)
        return _parse(self.cloud.generate(msgs, ticket=ticket, max_tokens=self.max_tokens).text)

    def compare(self, query: str, candidate: str, reference: str, request_id: str = "") -> PairwiseResult:
        ab = self._ask(query, candidate, reference, request_id)
        ba = self._ask(query, reference, candidate, request_id)
        score = {"A": 1, "TIE": 0, "B": -1}[ab] + {"B": 1, "TIE": 0, "A": -1}[ba]
        verdict = "win" if score > 0 else "loss" if score < 0 else "tie"
        return PairwiseResult(verdict, ab, ba)


def at_least_as_good_rate(results: list[PairwiseResult]) -> float:
    return sum(r.verdict in ("win", "tie") for r in results) / len(results) if results else 0.0


def is_empty_loss(record: dict) -> bool:
    return ("exact_match" not in record and not (record.get("answer") or "").strip()
            and not record.get("judge") and "judge_error" not in record)


def quality_summary(records: list[dict]) -> dict | None:
    verdicts = [r["judge"]["verdict"] for r in records if r.get("judge")]
    if not verdicts:
        return None
    empty = sum(map(is_empty_loss, records))
    return {"at_least_as_good": sum(v in ("win", "tie") for v in verdicts) / (len(verdicts) + empty),
            "verdicts": {v: verdicts.count(v) for v in ("win", "tie", "loss")},
            "empty_answers_as_loss": empty,
            "judge_errors": sum("judge_error" in r for r in records),
            "scoring": "empty-as-loss"}


NUMBER = r"-?\d+(?:\.\d+)?"
ANSWER_LINE = re.compile(r"^[\W_]*(?:final\s+)?answer[\s*_]*:[\s*_$]*(-?\d+(?:\.\d+)?)", re.I | re.M)


def answer_line_number(text: str) -> str | None:
    found = ANSWER_LINE.findall(text.split("</think>")[-1].replace(",", ""))
    return found[-1] if found else None


def extract_number(text: str) -> str | None:
    if answer_line_number(text) is not None:
        return answer_line_number(text)
    text = text.split("</think>")[-1].replace(",", "")
    low = text.lower()
    if "final answer" in low and re.search(NUMBER, text[low.rfind("final answer"):]):
        text = text[low.rfind("final answer"):]
    boxed = [b for b in re.findall(r"\\boxed\{([^{}]*)\}", text) if re.search(NUMBER, b)]
    if boxed:
        return re.findall(NUMBER, boxed[-1])[-1]
    bold = [b for b in re.findall(r"\*\*(.+?)\*\*", text, re.S) if re.search(NUMBER, b)]
    if bold:
        return re.findall(NUMBER, bold[-1])[-1]
    nums = re.findall(NUMBER, text)
    return nums[-1] if nums else None


def numeric_exact_match(prediction: str, gold: str) -> bool:
    p = extract_number(prediction)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gold)) < 1e-6
    except ValueError:
        return False


OPTION_ALIASES = {"chickenpox": "chicken pox", "gerd": "gastroesophageal reflux disease",
                  "acid reflux disease": "gastroesophageal reflux disease", "uti": "urinary tract infection",
                  "asthma": "bronchial asthma", "peptic ulcer": "peptic ulcer disease"}


def chosen_option(prediction: str, options: list[str]) -> str | None:
    text = re.sub(r"[-‐–]", " ", prediction.split("</think>")[-1].lower())
    names = {o.lower(): o for o in options} | {a: o for a, o in OPTION_ALIASES.items() if o in options}
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")

    def first(s):
        m = pattern.search(s)
        return names[m.group(1)] if m else None
    if "final answer" in text and first(text.rsplit("final answer", 1)[1]):
        return first(text.rsplit("final answer", 1)[1])
    for sentence in re.split(r"(?<=[.!?\n])", text):
        if "most likely" in sentence and first(sentence):
            return first(sentence)
    return first(text)


def option_exact_match(prediction: str, gold: str, options: list[str]) -> bool:
    return chosen_option(prediction, options) == gold


def answer_exact_match(prediction: str, gold: str, options: list[str] | None = None) -> bool:
    return option_exact_match(prediction, gold, options) if options else numeric_exact_match(prediction, gold)
