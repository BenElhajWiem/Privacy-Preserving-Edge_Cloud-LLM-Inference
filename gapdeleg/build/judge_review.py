from __future__ import annotations

import json
import sys
from collections import Counter

from gapdeleg import REPO_ROOT
from gapdeleg.build.batch import json_params, parse_json, run_batch
from gapdeleg.build.io import read_examples, write_examples
from gapdeleg.metrics.quality import JUDGE_PROMPT, _parse

ITEMS = REPO_ROOT / "data/judge_validation/items.jsonl"
REVIEW = REPO_ROOT / "data/review"
OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"


def items() -> list[dict]:
    return [json.loads(x) for x in ITEMS.read_text().splitlines()]


def opus():
    reqs = {}
    for it in items():
        for order, (a, b) in (("ab", (it["candidate"], it["reference"])), ("ba", (it["reference"], it["candidate"]))):
            reqs[f"{it['item']}-{order}"] = {
                "model": OPUS, "max_tokens": 4096, "output_config": {"effort": "low"},
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(query=it["query"], a=a, b=b)}]}
    run_batch("judge-validation-opus", reqs)


REF_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "issues", "note"],
              "properties": {
                  "verdict": {"type": "string", "enum": ["good", "acceptable", "poor"]},
                  "issues": {"type": "array", "items": {"type": "string", "enum": [
                      "wrong_language", "translation_instead_of_answer", "truncated", "off_task", "refusal",
                      "incomplete", "factually_wrong", "ignores_format"]}},
                  "note": {"type": "string"}}}

REF_PROMPT = """Below is a user's request to an AI assistant and the response it got. Rate how well the response
serves as a reference answer for evaluating other assistants on this request.

- good: does what was asked, in the language and format the request implies.
- acceptable: does the task with minor gaps.
- poor: does not do what was asked (for example answers in the wrong language, only translates when the user
  asked for something else, is cut off, refuses, or is off-task).
If the request itself asks for a translation, a translation is the right answer.

<request>
{query}
</request>

<response>
{response}
</response>"""


def refs():
    reqs = {ex.id: json_params(SONNET, REF_PROMPT.format(query=ex.query, response=ex.reference), REF_SCHEMA,
                               max_tokens=2048, effort="low")
            for ex in read_examples("pupa_tnb")}
    run_batch("pupa-tnb-reference-audit", reqs)


def report() -> dict:
    its = {it["item"]: it for it in items()}
    gpt = {json.loads(x)["item"]: json.loads(x)
           for x in (REPO_ROOT / "data/judge_validation/judge.jsonl").read_text().splitlines()}
    raw = {json.loads(x)["custom_id"]: json.loads(x)
           for x in (REPO_ROOT / "runs/batches/judge-validation-opus.jsonl").read_text().splitlines()}
    opus_v = {}
    for i in its:
        ab = _parse(raw[f"{i}-ab"].get("text", "")) if raw[f"{i}-ab"]["ok"] else "TIE"
        ba = _parse(raw[f"{i}-ba"].get("text", "")) if raw[f"{i}-ba"]["ok"] else "TIE"
        score = {"A": 1, "TIE": 0, "B": -1}[ab] + {"B": 1, "TIE": 0, "A": -1}[ba]
        opus_v[i] = {"verdict": "win" if score > 0 else "loss" if score < 0 else "tie", "order_ab": ab, "order_ba": ba}

    def position_bias(v):
        firsts = [x for d in v.values() for x in (d["order_ab"], d["order_ba"]) if x != "TIE"]
        return sum(x == "A" for x in firsts) / len(firsts) if firsts else None

    def consistent(v):
        return sum((d["order_ab"], d["order_ba"]) in (("A", "B"), ("B", "A"), ("TIE", "TIE")) for d in v.values())

    agree = sum(gpt[i]["verdict"] == opus_v[i]["verdict"] for i in its)
    by_sys = {}
    for name, v in (("gpt-5.5", gpt), (OPUS, opus_v)):
        for s in ("edge_only", "redact_send"):
            by_sys[f"{name} / {s}"] = dict(Counter(v[i]["verdict"] for i in its if its[i]["system"] == s))
    audit = {json.loads(x)["custom_id"]: parse_json(json.loads(x))
             for x in (REPO_ROOT / "runs/batches/pupa-tnb-reference-audit.jsonl").read_text().splitlines()}
    audited = {k: v for k, v in audit.items() if v}
    ref = {"audited": len(audited), "failed": len(audit) - len(audited),
           "verdicts": dict(Counter(v["verdict"] for v in audited.values())),
           "issues": dict(Counter(x for v in audited.values() for x in v["issues"]).most_common())}
    judge_items = {it["id"] for it in its.values()}
    ref["verdicts_in_judge_set"] = dict(Counter(audited[k]["verdict"] for k in judge_items if k in audited))
    out = {"judge_judge_agreement": agree / len(its), "n": len(its),
           "consistent_across_orders": {"gpt-5.5": consistent(gpt), OPUS: consistent(opus_v)},
           "position_bias_first_shown": {"gpt-5.5": position_bias(gpt), OPUS: position_bias(opus_v)},
           "verdicts_by_system": by_sys, "reference_audit": ref,
           "per_item": {i: {"system": its[i]["system"], "id": its[i]["id"], "gpt": gpt[i]["verdict"],
                            "opus": opus_v[i]["verdict"],
                            "ref_quality": (audited.get(its[i]["id"]) or {}).get("verdict")} for i in its}}
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "judge_review.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def newrefs():
    targets = [("pupa_tnb", None), ("pupa_new", "dev")]
    reqs = {}
    for name, split in targets:
        for ex in read_examples(name):
            if split is None or ex.meta.get("split") == split:
                reqs[ex.id] = {"model": SONNET, "max_tokens": 8192, "output_config": {"effort": "medium"},
                               "messages": [{"role": "user", "content": ex.query}]}
    res = run_batch("pupa-cloud-full-references", reqs)
    for name, _ in targets:
        exs = read_examples(name)
        for ex in exs:
            r = res.get(ex.id)
            if r and r["ok"] and r.get("text") and r.get("stop_reason") == "end_turn":
                ex.meta.setdefault("pupa_reference", ex.reference)
                ex.reference = r["text"]
                ex.meta["reference_source"] = f"cloud_full:{SONNET}"
            elif r:
                ex.meta["reference_source"] = f"pupa (cloud_full failed: {r.get('stop_reason') or r.get('error')})"
        write_examples(name, exs)


def v2set():
    import random
    from dataclasses import asdict

    from gapdeleg.backends import make_cloud
    from gapdeleg.build.batch import cost, log_cost
    from gapdeleg.metrics.quality import PairwiseJudge
    from gapdeleg.privacy.guard import EgressGuard
    refs = {ex.id: ex for ex in read_examples("pupa_tnb")}
    rng = random.Random(22)
    out = []
    for it in items():
        ex = refs[it["id"]]
        out.append({**it, "reference": ex.reference, "reference_source": ex.meta.get("reference_source", "pupa"),
                    "candidate_is_a": rng.random() < 0.5})
    with open(REPO_ROOT / "data/judge_validation/items_v2.jsonl", "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    judge_model = make_cloud({"backend": "openai"}, cache_dir=REPO_ROOT / "runs/cache")
    judge = PairwiseJudge(judge_model, EgressGuard())
    with open(REPO_ROOT / "data/judge_validation/judge_v2.jsonl", "w") as f:
        for it in out:
            r = judge.compare(it["query"], it["candidate"], it["reference"], request_id=it["id"])
            f.write(json.dumps({"item": it["item"], "id": it["id"], "judge_model": judge_model.name,
                                "verdict": r.verdict, "order_ab": r.order_ab, "order_ba": r.order_ba}) + "\n")
    usage = asdict(judge_model.spent)
    log_cost("judge-validation-v2", judge_model.name, usage, cost(judge_model.name, usage, batch=False),
             2 * len(out), batch=False)
    v = [json.loads(x) for x in (REPO_ROOT / "data/judge_validation/judge_v2.jsonl").read_text().splitlines()]
    sysof = {it["item"]: it["system"] for it in out}
    print({s: dict(Counter(x["verdict"] for x in v if sysof[x["item"]] == s)) for s in ("edge_only", "redact_send")})


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "env"))
    from envfile import load_env
    load_env()
    cmd = sys.argv[1]
    if cmd == "report":
        r = report()
        r.pop("per_item")
        print(json.dumps(r, indent=1))
    else:
        {"opus": opus, "refs": refs, "newrefs": newrefs, "v2set": v2set}[cmd]()
