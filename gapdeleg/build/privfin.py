from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from gapdeleg.build.align import all_occurrences
from gapdeleg.build.batch import json_params, parse_json, run_batch
from gapdeleg.build.io import PROCESSED, write_examples
from gapdeleg.build.personas import make_persona
from gapdeleg.data.loaders import load_gsm8k
from gapdeleg.data.schema import Example
from gapdeleg.metrics.quality import numeric_exact_match
from gapdeleg.privacy.types import Label, Span

GEN_MODEL = "claude-sonnet-5-5"
SOLVE_MODEL = "claude-opus-5-5"
SEED = 7
TARGETS = {"test": 300, "dev": 60}
CANDIDATES = {"test": 360, "dev": 75}
ANSWER_LINE = 'Please finish your reply with a last line that starts with "Answer:" followed by the final number.'
CUES = re.compile(r"\$|years? old|\bage\b|\b(?:January|February|March|April|May|June|July|August|September|"
                  r"October|November|December)\b|\bweeks?\b|\bmonths?\b|\bhours?\b|\bsalary\b|\bsaves?\b", re.I)
NUM = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")

GEN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rewritten"],
              "properties": {"rewritten": {"type": "string"}}}
SOLVE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
                "properties": {"answer": {"type": "number"}}}

GEN_PROMPT = """Rewrite this math word problem as a first-person message from a real person asking for help
with their own situation. You are this person:

- name: {name}
- employer: {employer}
- bank and account: {bank}, account number {account}

Rules:
- Mention the full name "{name}", the employer "{employer}", the bank "{bank}" and the account number
  "{account}" naturally, each copied exactly.
- Keep every quantity and relation of the original so that the answer is exactly the same number. Write
  every quantity in digits (for example "3", not "three"), and keep each number from the original as written.
- Map the problem's people and items onto the persona's life where it reads naturally (the money can be
  theirs, the purchases their own), but do not change any value.
- Do not add any other numbers apart from the account number.
- End with a question that asks for the same quantity as the original.

Original problem:
{question}"""

SOLVE_PROMPT = """Solve this problem and give only the final numeric answer.

{question}"""


def numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in NUM.findall(text)]


def candidates(split: str) -> list[Example]:
    pool = [e for e in load_gsm8k("test" if split == "test" else "train") if CUES.search(e.query)]
    rng = random.Random(SEED + (0 if split == "test" else 1))
    rng.shuffle(pool)
    return pool[:CANDIDATES[split]]


def persona_for(ex: Example):
    return make_persona(int(ex.id.rsplit("-", 1)[1]) * 7919 + (0 if "test" in ex.id else 1))


def check_rewrite(original: str, rewritten: str, p) -> list[str]:
    problems = [f"missing {k}" for k, v in (("name", p.name), ("employer", p.employer), ("bank", p.bank),
                                            ("account", p.account)) if v not in rewritten]
    have = Counter(numbers(rewritten.replace(p.account, " ")))
    need = Counter(numbers(original))
    for n, c in need.items():
        if have[n] < 1:
            problems.append(f"number {n} missing")
    return problems


def spans_for(rewritten: str, p) -> list[Span]:
    spans: list[Span] = []

    def free(s, e):
        return all(e <= x.start or s >= x.end for x in spans)
    for text, cat in ((p.name, "PERSON"), (p.employer, "ORGANIZATION"), (p.bank, "ORGANIZATION"),
                      (p.account, "ACCOUNT_NUMBER"), (p.first, "PERSON"), (p.last, "PERSON")):
        for s, e in all_occurrences(rewritten, text):
            if free(s, e):
                spans.append(Span(rewritten[s:e], Label.LOCAL_ONLY, cat, s, e, "persona"))
    for m in NUM.finditer(rewritten):
        if not free(m.start(), m.end()):
            continue
        money = m.start() > 0 and rewritten[m.start() - 1] == "$"
        s = m.start() - 1 if money else m.start()
        spans.append(Span(rewritten[s:m.end()], Label.LOCAL_ONLY, "MONEY" if money else "AMOUNT", s, m.end(),
                          "quantity"))
    return sorted(spans, key=lambda x: x.start)


REVIEW_DROPS = {"gsm8k-test-666"}
REALISM = {"disclaimer": re.compile(r"nothing to do with|not what I.m asking|separate matter|unrelated to|irrelevant|"
                                    r"if that matters|not relevant", re.I),
           "puzzle_framing": re.compile(r"puzzle|word problem|math problem|homework|riddle", re.I)}

NUMBER_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                          r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                          r"eighty|ninety|hundred|thousand|dozen)\b", re.I)


def augment_spans(ex: Example) -> Example:
    nlp = _nlp()
    spans = list(ex.private_spans)

    def free(s, e):
        return all(e <= x.start or s >= x.end for x in spans)
    p = ex.meta["persona"]
    for ent in nlp(ex.query).ents:
        if ent.label_ == "PERSON" and ent.text in ex.meta["original"] and p["first"] not in ent.text \
                and free(ent.start_char, ent.end_char):
            for s, e in all_occurrences(ex.query, ent.text):
                if free(s, e):
                    spans.append(Span(ex.query[s:e], Label.LOCAL_ONLY, "PERSON", s, e, "gsm8k-name"))
    for m in NUMBER_WORDS.finditer(ex.query):
        if free(m.start(), m.end()):
            spans.append(Span(m.group(0), Label.LOCAL_ONLY, "AMOUNT", m.start(), m.end(), "quantity-word"))
    ex.private_spans = sorted(spans, key=lambda x: x.start)
    return ex


_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_lg")
    return _NLP


def build(split: str) -> tuple[list[Example], dict]:
    cands = candidates(split)
    gen = run_batch(f"privfin-gen-{split}", {
        e.id: json_params(GEN_MODEL, GEN_PROMPT.format(question=e.query, **persona_for(e).to_dict()), GEN_SCHEMA,
                          max_tokens=4096, effort="low") for e in cands})
    rewritten, drops = {}, Counter()
    for e in cands:
        data = parse_json(gen.get(e.id))
        if data is None:
            drops["generation failed"] += 1
            continue
        text = data["rewritten"].strip()
        problems = check_rewrite(e.query, text, persona_for(e))
        if problems:
            drops["rewrite check: " + ("number missing" if problems[0].startswith("number") else problems[0])] += 1
            continue
        rewritten[e.id] = text
    solved = run_batch(f"privfin-verify-{split}", {
        i: json_params(SOLVE_MODEL, SOLVE_PROMPT.format(question=t), SOLVE_SCHEMA, max_tokens=8192, effort="low")
        for i, t in rewritten.items()})
    out = []
    for e in cands:
        if e.id not in rewritten:
            continue
        data = parse_json(solved.get(e.id))
        if data is None:
            drops["solver failed or refused"] += 1
            continue
        ans = data["answer"]
        if not numeric_exact_match(str(ans), e.reference):
            drops["solver answer != gold"] += 1
            continue
        if e.id in REVIEW_DROPS:
            drops["dropped in review"] += 1
            continue
        if len(out) >= TARGETS[split]:
            drops["over target"] += 1
            continue
        p = persona_for(e)
        text = rewritten[e.id]
        gsm_index = e.id.rsplit("-", 1)[1]
        out.append(augment_spans(Example(id=f"privfin-{split}-gsm{gsm_index}", query=f"{text}\n\n{ANSWER_LINE}",
                                         reference=e.reference,
                           private_spans=spans_for(text, p), family="B",
                           meta={"source_id": e.id, "original": e.query, "solution": e.meta["solution"],
                                 "persona": p.to_dict(), "split": split, "gen_model": GEN_MODEL,
                                 "verify_model": SOLVE_MODEL, "verify_answer": ans,
                                 "realism_flags": [k for k, rx in REALISM.items() if rx.search(text)]})))
    stats = {"candidates": len(cands), "kept": len(out), "dropped": dict(drops),
             "spans": sum(len(x.private_spans) for x in out),
             "by_category": dict(Counter(s.category for x in out for s in x.private_spans).most_common())}
    return out, stats


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "env"))
    from envfile import load_env
    load_env()
    stats = {}
    for split in ("test", "dev"):
        exs, st = build(split)
        write_examples(f"privfin_{split}", exs)
        stats[f"privfin_{split}"] = st
    (PROCESSED / "privfin_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
