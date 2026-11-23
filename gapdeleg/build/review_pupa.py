from __future__ import annotations

import json
import re
import sys
from collections import Counter

from gapdeleg import REPO_ROOT
from gapdeleg.build.batch import json_params, parse_json, run_batch
from gapdeleg.build.io import read_examples, write_examples
from gapdeleg.build.openai_batch import chat_json_params, run_openai_batch
from gapdeleg.build.pupa_labels import CATEGORIES, LABELS, POLICY
from gapdeleg.privacy.matching import normalize
from gapdeleg.privacy.types import Label

MODEL = "gpt-5.5"
REVIEW = REPO_ROOT / "data/review"

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["labels"],
          "properties": {"labels": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["id", "label", "category"],
              "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": LABELS},
                             "category": {"type": "string", "enum": CATEGORIES}}}}}}

PROMPT = """Label each numbered item of private information in this request under the policy.

<request>
{query}
</request>

Items (copied from the request):
{items}

Return one entry per item id."""


def distinct_spans(ex) -> list[str]:
    seen, out = set(), []
    for s in ex.private_spans:
        k = normalize(s.text)
        if k and k not in seen:
            seen.add(k)
            out.append(s.text)
    return out


def second():
    reqs = {}
    for name in ("pupa_tnb", "pupa_new"):
        for ex in read_examples(name):
            items = "\n".join(f"{i}: {t}" for i, t in enumerate(distinct_spans(ex)))
            if items:
                reqs[ex.id] = chat_json_params(MODEL, POLICY, PROMPT.format(query=ex.query, items=items), SCHEMA,
                                               name="labels")
    run_openai_batch("pupa-second-annotator", reqs)


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(write: bool = True) -> dict:
    res = {json.loads(x)["custom_id"]: json.loads(x)
           for x in (REPO_ROOT / "runs/batches/pupa-second-annotator.jsonl").read_text().splitlines()}
    rows, missing = [], 0
    for name in ("pupa_tnb", "pupa_new"):
        for ex in read_examples(name):
            texts = distinct_spans(ex)
            if not texts:
                continue
            data = parse_json(res.get(ex.id)) or {"labels": []}
            second = {d["id"]: d for d in data["labels"]}
            first = {normalize(s.text): s for s in ex.private_spans}
            for i, t in enumerate(texts):
                s = first[normalize(t)]
                if i not in second:
                    missing += 1
                    continue
                rows.append({"dataset": name, "id": ex.id, "span": t, "start": s.start, "end": s.end,
                             "sonnet": s.label.value, "sonnet_cat": s.category, "source": s.source,
                             "gpt": second[i]["label"], "gpt_cat": second[i]["category"],
                             "first_is_default": s.source.startswith("pupa-unit-unlabeled")
                             or ex.meta.get("label_status", "ok") != "ok"})
    pairs = [(r["sonnet"], r["gpt"]) for r in rows]
    by_set = {}
    for name in ("pupa_tnb", "pupa_new", None):
        sub = [r for r in rows if name is None or r["dataset"] == name]
        p = [(r["sonnet"], r["gpt"]) for r in sub]
        by_set[name or "all"] = {"spans": len(p), "agreement": sum(a == b for a, b in p) / len(p), "kappa": kappa(p)}
    per_label = {lab: {"sonnet_n": sum(a == lab for a, _ in pairs), "gpt_n": sum(b == lab for _, b in pairs),
                       "both": sum(a == b == lab for a, b in pairs)} for lab in LABELS}
    confusion = Counter(f"{a} -> {b}" for a, b in pairs if a != b)
    stats = {"model": MODEL, "missing_from_second": missing, "by_set": by_set, "per_label": per_label,
             "disagreements": dict(confusion.most_common()), "n_disagreements": sum(confusion.values())}
    if write:
        REVIEW.mkdir(parents=True, exist_ok=True)
        with open(REVIEW / "pupa_disagreements.jsonl", "w") as f:
            for r in rows:
                if r["sonnet"] != r["gpt"] or r["first_is_default"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (REVIEW / "pupa_agreement.json").write_text(json.dumps(stats, indent=1))
    return stats


ADJ_MODEL = "claude-opus-5-5"
ADJ_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decisions"],
              "properties": {"decisions": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False, "required": ["id", "label", "category", "reason"],
                  "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": LABELS},
                                 "category": {"type": "string", "enum": CATEGORIES},
                                 "reason": {"type": "string"}}}}}}
ADJ_PROMPT = """Two annotators labeled private information in this request under the policy and disagreed on the
items below. Decide the correct label for each item under the policy, including the clarifications.
Give a one-sentence reason.

<request>
{query}
</request>

Disputed items:
{items}"""
STRICT = {"public": 0, "abstractable": 1, "local-only": 2}

REVIEWER_OVERRIDES = {
    ("pupa-new-60", "jonah van beijnen"): ("local-only", "Asked about as a person; nothing shows a public "
                                                         "profile, so the doubt rule applies."),
    ("pupa-new-27", "Xijing Hospital"): ("abstractable", "Study site in the user's own unpublished abstract "
                                                         "(policy rule 9, drafts of own work)."),
    ("pupa-new-83", "Xijing Hospital"): ("abstractable", "Study site in the user's own unpublished abstract "
                                                         "(policy rule 9, drafts of own work)."),
}

PLACEHOLDER = re.compile(r"^<?PRESIDIO_ANONYMIZED_[A-Z_]+>?$")
PRONOUN_FIXES = {("pupa-new-360", "She"), ("pupa-new-360", "her")}


def fix_artifacts() -> dict:
    fixed: Counter = Counter()
    for name in ("pupa_tnb", "pupa_new"):
        exs = read_examples(name)
        for ex in exs:
            spans = []
            for s in ex.private_spans:
                if PLACEHOLDER.match(s.text.strip()) or (ex.id, s.text) in PRONOUN_FIXES:
                    if s.label != Label.PUBLIC:
                        fixed[f"{name}: {s.label.value} -> public"] += 1
                    s = type(s)(**{**s.__dict__, "label": Label.PUBLIC, "provenance": "artifact-fix"})
                spans.append(s)
            ex.private_spans = spans
        write_examples(name, exs)
    return dict(fixed)


def disputed() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for x in (REVIEW / "pupa_disagreements.jsonl").read_text().splitlines():
        r = json.loads(x)
        out.setdefault(r["id"], []).append(r)
    return out


def adjudicate():
    queries = {ex.id: ex.query for n in ("pupa_tnb", "pupa_new") for ex in read_examples(n)}
    reqs = {}
    for rid, rows in disputed().items():
        items = "\n".join(
            f'{i}: "{r["span"]}" (annotator A: {"none" if r["first_is_default"] else r["sonnet"]}; '
            f'annotator B: {r["gpt"]})' for i, r in enumerate(rows))
        reqs[rid] = json_params(ADJ_MODEL, ADJ_PROMPT.format(query=queries[rid], items=items), ADJ_SCHEMA,
                                system=POLICY, max_tokens=8192, effort="medium")
    run_batch("pupa-adjudication", reqs)


def finalize() -> dict:
    res = {json.loads(x)["custom_id"]: json.loads(x)
           for x in (REPO_ROOT / "runs/batches/pupa-adjudication.jsonl").read_text().splitlines()}
    disp = disputed()
    prov_counts: Counter = Counter()
    changes: Counter = Counter()
    for name in ("pupa_tnb", "pupa_new"):
        exs = read_examples(name)
        for ex in exs:
            rows = disp.get(ex.id, [])
            dec = {d["id"]: d for d in (parse_json(res.get(ex.id)) or {"decisions": []})["decisions"]}
            final = {}
            for i, r in enumerate(rows):
                d = dec.get(i)
                if d:
                    final[normalize(r["span"])] = (d["label"], d["category"], "adjudicated", d["reason"], r)
                else:
                    lab = max(r["sonnet"], r["gpt"], key=STRICT.get)
                    final[normalize(r["span"])] = (lab, r["sonnet_cat"], "strict-default", "", r)
                if (ex.id, r["span"]) in REVIEWER_OVERRIDES:
                    lab, why = REVIEWER_OVERRIDES[(ex.id, r["span"])]
                    final[normalize(r["span"])] = (lab, final[normalize(r["span"])][1], "reviewer", why, r)
            review = {}
            new_spans = []
            for s in ex.private_spans:
                f = final.get(normalize(s.text))
                if f is None:
                    new_spans.append(type(s)(**{**s.__dict__, "provenance": "agreed"}))
                    prov_counts["agreed"] += 1
                    continue
                lab, cat, prov, reason, r = f
                if lab != r["sonnet"]:
                    changes[f"{r['sonnet']} -> {lab}"] += 1
                new_spans.append(type(s)(**{**s.__dict__, "label": Label(lab), "category": cat, "provenance": prov}))
                prov_counts[prov] += 1
                review[s.text] = {"sonnet": r["sonnet"], "gpt": r["gpt"], "final": lab, "provenance": prov,
                                  "reason": reason}
            ex.private_spans = new_spans
            ex.meta["label_review"] = review
            ex.meta["label_policy"] = "v1.1 (labels: Sonnet under v1.0, GPT-5.5 second annotator, Opus adjudication)"
        write_examples(name, exs)
    out = {"span_provenance": dict(prov_counts), "label_changes_from_sonnet": dict(changes.most_common())}
    (REVIEW / "pupa_adjudication.json").write_text(json.dumps(out, indent=1))
    out["artifact_fix"] = fix_artifacts()
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "env"))
    from envfile import load_env
    load_env()
    {"second": second, "agreement": lambda: print(json.dumps(agreement(), indent=1)), "adjudicate": adjudicate,
     "finalize": lambda: print(json.dumps(finalize(), indent=1)),
     "fix-artifacts": lambda: print(json.dumps(fix_artifacts(), indent=1))}[sys.argv[1]]()
