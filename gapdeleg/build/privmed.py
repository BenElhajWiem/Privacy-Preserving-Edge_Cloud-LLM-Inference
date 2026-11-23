from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter

from gapdeleg import REPO_ROOT
from gapdeleg.build.align import align, all_occurrences
from gapdeleg.build.batch import json_params, parse_json, run_batch
from gapdeleg.build.io import PROCESSED, write_examples
from gapdeleg.build.personas import make_persona
from gapdeleg.data.loaders import EXTERNAL, load_prism
from gapdeleg.data.schema import Example
from gapdeleg.privacy.types import Label, Span

PRISM_RE = re.compile(r"A (\d+)-year-old (\w+) patient named (\w+) reports symptoms: (.+)\. (What .+\?)$")
GRETEL = EXTERNAL / "symptom_to_diagnosis"
SEED = 11
SPAN_MODEL, SOLVE_MODEL = "claude-sonnet-5-5", "claude-opus-5-5"

ADULT = ("Hi, I'm {name}, a {age}-year-old {sex}. I work at {employer} in {city}. Lately I've been dealing "
         "with {symptoms}. My insurer is {insurer} (member ID {member_id}) and my number is {phone}. {question}")
CHILD = ("Hi, I'm {name} and I work at {employer} in {city}. My {age}-year-old {child_word}, {child}, has been "
         "dealing with {symptoms}. {child_pron} is on my insurance with {insurer} (member ID {member_id}) and "
         "you can reach me at {phone}. {question}")
GRETEL_T = ("Hi, I'm {name}, {age}, and I work at {employer} in {city}. {description} My insurer is {insurer} "
            "(member ID {member_id}) and my number is {phone}. Which of these is the most likely diagnosis: "
            "{options}? Please name one.")

SYMPTOM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["symptoms"],
                  "properties": {"symptoms": {"type": "array", "items": {"type": "string"}}}}
SYMPTOM_PROMPT = """List every phrase in this patient's message that describes a symptom, sign or health complaint.
Copy each phrase exactly as it appears (same words and spelling), as short as possible while still meaning
the symptom, for example "a burning sensation in my stomach" or "heartburn".

Message:
{text}"""


def symptom_text(s: str) -> str:
    return s.replace("_/_", " or ").replace("_", " ").strip()


def add_spans(text: str, fields) -> list[Span]:
    spans: list[Span] = []
    for value, cat, label, source in fields:
        hits = all_occurrences(text, value)
        if not hits and (a := align(text, value)) and a[2] == "exact":
            hits = [a[:2]]
        for s, e in hits:
            if all(e <= x.start or s >= x.end for x in spans):
                spans.append(Span(text[s:e], label, cat, s, e, source))
    return sorted(spans, key=lambda x: x.start)


def persona_fields(p, age_text: str) -> list:
    return [(p.name, "PERSON", Label.LOCAL_ONLY, "persona"), (p.first, "PERSON", Label.LOCAL_ONLY, "persona"),
            (p.employer, "ORGANIZATION", Label.LOCAL_ONLY, "persona"),
            (p.member_id, "ID_NUMBER", Label.LOCAL_ONLY, "persona"),
            (p.phone, "PHONE_NUMBER", Label.LOCAL_ONLY, "persona"),
            (age_text, "AGE", Label.ABSTRACTABLE, "persona"), (p.city, "LOCATION", Label.ABSTRACTABLE, "persona"),
            (p.insurer, "ORGANIZATION", Label.ABSTRACTABLE, "persona")]


def generalized(p, age: int) -> dict:
    return {"age": f"in their {age // 10 * 10}s" if age >= 20 else "a child", p.city: "a European city",
            p.insurer: "a health insurer"}


def build_prism() -> tuple[list[Example], dict]:
    out, fixes = [], Counter()
    for i, ex in enumerate(load_prism("Medical")):
        age, sex, prism_name, symptoms, question = PRISM_RE.match(ex.query).groups()
        age = int(age)
        rng = random.Random(SEED * 1000 + i)
        orig_age = age
        if age > 95:
            age = rng.randint(60, 95)
            fixes["age resampled (>95)"] += 1
        syms = [symptom_text(s) for s in symptoms.split(", ")]
        sym_str = ", ".join(syms[:-1]) + (" and " if len(syms) > 1 else "") + syms[-1]
        if age < 18:
            fixes["minor: written by a parent"] += 1
            p = make_persona(100_000 + i, sex=rng.choice(["female", "male"]))
            child = make_persona(200_000 + i, sex=sex).first
            text = CHILD.format(name=p.name, employer=p.employer, city=p.city, age=age,
                                child_word="daughter" if sex == "female" else "son", child=child,
                                symptoms=sym_str, child_pron="She" if sex == "female" else "He",
                                insurer=p.insurer, member_id=p.member_id, phone=p.phone, question=question)
            fields = persona_fields(p, f"{age}-year-old") + [(child, "PERSON", Label.LOCAL_ONLY, "persona-child")]
        else:
            p = make_persona(100_000 + i, sex=sex)
            text = ADULT.format(name=p.name, age=age, sex=sex, employer=p.employer, city=p.city, symptoms=sym_str,
                                insurer=p.insurer, member_id=p.member_id, phone=p.phone, question=question)
            fields = persona_fields(p, f"{age}-year-old")
        fields += [(s, "MEDICAL_CONDITION", Label.LOCAL_ONLY, "prism") for s in syms]
        out.append(Example(id=f"privmed-prism-{i}", query=text, reference=None, family="B",
                           private_spans=add_spans(text, fields),
                           meta={"source": "prism", "source_id": ex.id, "original": ex.query,
                                 "prism_name": prism_name, "original_age": orig_age, "age": age, "sex": sex,
                                 "symptoms": syms, "persona": p.to_dict(), "split": "prism",
                                 "generalized": generalized(p, age)}))
    return out, {"examples": len(out), "fixes": dict(fixes)}


def gretel_rows(split: str) -> list[dict]:
    rows = [json.loads(x) for x in (GRETEL / f"{'test' if split == 'test' else 'train'}.jsonl").read_text().splitlines()]
    if split == "dev":
        rows = random.Random(SEED).sample(rows, 40)
    return rows


def options() -> list[str]:
    return sorted({json.loads(x)["output_text"] for x in (GRETEL / "train.jsonl").read_text().splitlines()})


def build_gretel(split: str) -> tuple[list[Example], dict]:
    opts = options()
    texts = {}
    meta = {}
    for i, r in enumerate(gretel_rows(split)):
        rng = random.Random(SEED * 7919 + i + (0 if split == "test" else 10_000))
        p = make_persona(300_000 + i + (0 if split == "test" else 10_000))
        age = rng.randint(18, 85)
        desc = r["input_text"].strip()
        text = GRETEL_T.format(name=p.name, age=age, employer=p.employer, city=p.city, description=desc,
                               insurer=p.insurer, member_id=p.member_id, phone=p.phone, options=", ".join(opts))
        eid = f"privmed-{split}-{i}"
        texts[eid] = text
        meta[eid] = (p, age, desc, r["output_text"])
    found = run_batch(f"privmed-symptoms-{split}", {
        k: json_params(SPAN_MODEL, SYMPTOM_PROMPT.format(text=meta[k][2]), SYMPTOM_SCHEMA, max_tokens=2048)
        for k in texts})
    solve_schema = {"type": "object", "additionalProperties": False, "required": ["diagnosis"],
                    "properties": {"diagnosis": {"type": "string", "enum": opts}}}
    solved = run_batch(f"privmed-verify-{split}", {
        k: json_params(SOLVE_MODEL, t, solve_schema, max_tokens=4096) for k, t in texts.items()})
    out, st = [], Counter()
    for k, text in texts.items():
        p, age, desc, gold = meta[k]
        syms = (parse_json(found.get(k)) or {}).get("symptoms", [])
        sym_fields = [(s, "MEDICAL_CONDITION", Label.LOCAL_ONLY, "symptom-span") for s in syms if s.strip()]
        spans = add_spans(text, persona_fields(p, str(age)) + sym_fields)
        n_sym = sum(s.category == "MEDICAL_CONDITION" for s in spans)
        st["symptom spans"] += n_sym
        st["symptom phrases not found in text"] += sum(not all_occurrences(text, f[0]) and not align(text, f[0])
                                                        for f in sym_fields)
        if n_sym == 0:
            st["items with no symptom span"] += 1
        ans = (parse_json(solved.get(k)) or {}).get("diagnosis")
        st["opus correct"] += ans == gold
        out.append(Example(id=k, query=text, reference=gold, family="B", private_spans=spans,
                           meta={"source": "gretelai/symptom_to_diagnosis", "description": desc, "age": age,
                                 "options": opts, "persona": p.to_dict(), "split": split,
                                 "verify_model": SOLVE_MODEL, "verify_answer": ans,
                                 "generalized": generalized(p, age)}))
    st = dict(st)
    st["examples"] = len(out)
    st["opus_accuracy"] = st.get("opus correct", 0) / len(out)
    st["diagnoses"] = len(set(e.reference for e in out))
    return out, st


def label_stats(exs: list[Example]) -> dict:
    return {"spans": sum(len(x.private_spans) for x in exs),
            "by_label": dict(Counter(s.label.value for x in exs for s in x.private_spans)),
            "by_category": dict(Counter(s.category for x in exs for s in x.private_spans).most_common())}


def main():
    sys.path.insert(0, str(REPO_ROOT / "env"))
    from envfile import load_env
    load_env()
    stats = {}
    exs, st = build_prism()
    write_examples("privmed_prism", exs)
    stats["privmed_prism"] = {**st, **label_stats(exs)}
    for split in ("test", "dev"):
        exs, st = build_gretel(split)
        write_examples(f"privmed_{split}", exs)
        stats[f"privmed_{split}"] = {**st, **label_stats(exs)}
    (PROCESSED / "privmed_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
