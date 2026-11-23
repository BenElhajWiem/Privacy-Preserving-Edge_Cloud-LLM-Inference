#!/usr/bin/env python
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.baselines import MASK_INSTRUCTION
from gapdeleg.build.io import read_examples
from gapdeleg.metrics import leakage as leak_m
from gapdeleg.metrics.bootstrap import point_values, request_outcomes
from gapdeleg.privacy.matching import whole_word
from gapdeleg.privacy.policy import Policy

SPLITS = {"dev": "runs/pupa-quality-dev/curves", "pilot": "runs/pupa-quality-pilot/curves",
          "test": "runs/pupa-test/method/curves"}
DEV = "runs/pupa-quality-dev/curves/{}/selected.json"
MODELS = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b-q4_K_M": "Original Qwen3-4B",
          "qwen3-4b-instruct-2507": "Qwen3-4B-Instruct-2507", "llama3.2-3b": "Llama-3.2-3B", "gemma3-4b": "Gemma-3-4B"}
PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")
WORD = re.compile(r"\w+")
CHECKS = ("known value", "detector", "request check", "leak check", "error")
DRAWS = 10_000


def auroc(scores, labels) -> float | None:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def ece(conf, labels, bins: int = 10) -> float | None:
    c, y = np.asarray(conf, float), np.asarray(labels, float)
    if not len(c):
        return None
    idx = np.clip((c * bins).astype(int), 0, bins - 1)
    return float(sum(abs(c[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(c))


def masked_stretches(query: str, masked: str) -> list[tuple[int, int]] | None:
    parts = PLACEHOLDER.split(masked)
    if len(parts) == 1:
        return [] if masked == query else None
    if not query.startswith(parts[0]) or not query.endswith(parts[-1]):
        return None
    end = len(query) - len(parts[-1])
    start, out = len(parts[0]), []
    for lit in parts[1:-1]:
        if not lit:
            continue
        j = query.find(lit, start + 1)
        if j < 0 or j + len(lit) > end:
            return None
        out.append((start, j))
        start = j + len(lit)
    if end <= start:
        return None
    out.append((start, end))
    return out


def occurrences(query: str, text: str) -> list[tuple[int, int]]:
    t = text.strip()
    return [m.span() for m in re.finditer(whole_word(re.escape(t), True), query, re.I)] if t else []


def overlaps(a: tuple[int, int], spans) -> bool:
    return any(a[0] < e and s < a[1] for s, e in spans)


def check_of(reason: str) -> str:
    for prefix, name in (("local-only span", "known value"), ("detector:", "detector"),
                         ("request check", "request check"), ("local model", "leak check")):
        if reason.startswith(prefix):
            return name
    return "error"


def ratio(num, den):
    return num / den if den else None


def load_results(run: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (ROOT / "runs" / run / "results.jsonl").read_text().splitlines())}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def gate(cur: dict, threshold: float, rng: np.random.Generator) -> dict:
    op, cl = request_outcomes(ROOT / "runs" / cur["open_run"]), request_outcomes(ROOT / "runs" / cur["closed_run"])
    res = load_results(cur["open_run"])
    ids = [i for i in res if i in cl]
    conf = {i: res[i]["notes"].get("confidence", 0.0) for i in ids}

    def routed(send) -> list[dict]:
        return [dict(op[i], escalated=1) if send(i) else dict(cl[i], escalated=0) for i in ids]

    def summary(rows) -> dict:
        v = point_values(rows)
        return {"sent": float(np.mean([r["escalated"] for r in rows])), "quality": v["at_least_as_good"],
                "hard": v["hard_violation"], "local_leaked": v["span_leak_local_only"]}

    gate_rows = routed(lambda i: conf[i] < threshold)
    out = {"requests": len(ids), "threshold": threshold, "gate": summary(gate_rows),
           "send_all": summary(routed(lambda i: True)), "send_none": summary(routed(lambda i: False)),
           "oracle": summary(routed(lambda i: op[i]["good"] > cl[i]["good"]))}
    s = out["gate"]["sent"]
    exp = {k: sum(s * op[i][k] + (1 - s) * cl[i][k] for i in ids) for k in ("good", "judged", "hard")}
    out["random"] = {"sent": s, "quality": ratio(exp["good"], exp["judged"]), "hard": exp["hard"] / len(ids)}

    judged = [i for i in ids if op[i]["judged"] and cl[i]["judged"]]
    c = [conf[i] for i in judged]
    cloud_better = [op[i]["good"] > cl[i]["good"] for i in judged]
    edge_good = [bool(cl[i]["good"]) for i in judged]
    out.update(judged_both=len(judged), cloud_better=int(sum(cloud_better)), edge_good=int(sum(edge_good)),
               auroc_cloud_better=auroc([-x for x in c], cloud_better), auroc_edge_good=auroc(c, edge_good),
               ece_edge_good=ece(c, edge_good), mean_confidence=float(np.mean(c)) if c else None)

    cuts = sorted({0.0, *conf.values(), 1.01})
    equal = [summary(routed(lambda i, t=t: conf[i] < t)) | {"threshold": t} for t in cuts]
    target = out["send_all"]["quality"] or 0.0
    hits = [p for p in equal if (p["quality"] or 0.0) >= target - 1e-12]
    out["equal_quality"] = min(hits, key=lambda p: p["sent"]) if hits else None

    all_rows = routed(lambda i: True)
    g = {k: np.array([r[k] for r in gate_rows], float) for k in ("good", "judged", "hard")}
    a = {k: np.array([r[k] for r in all_rows], float) for k in ("good", "judged", "hard")}
    idx = rng.integers(0, len(ids), size=(DRAWS, len(ids)))
    dq = g["good"][idx].sum(1) / g["judged"][idx].sum(1) - a["good"][idx].sum(1) / a["judged"][idx].sum(1)
    dh = g["hard"][idx].mean(1) - a["hard"][idx].mean(1)
    out["gate_minus_send_all"] = {
        "quality": (out["gate"]["quality"] or 0) - (out["send_all"]["quality"] or 0),
        "quality_ci": [float(np.percentile(dq, 2.5)), float(np.percentile(dq, 97.5))],
        "hard": out["gate"]["hard"] - out["send_all"]["hard"],
        "hard_ci": [float(np.percentile(dh, 2.5)), float(np.percentile(dh, 97.5))]}
    return out


def guard(cur: dict, examples: dict, policy: Policy) -> dict:
    copy = ROOT / "runs" / cur["open_run"]
    orig = ROOT / "runs" / cur["open_run"].removesuffix("-no_gap_loop")
    res = load_results(orig.name)
    log = defaultdict(list)
    for x in load_jsonl(orig / "guard.jsonl"):
        if x["purpose"] == "delegation":
            log[x["request_id"]].append(x)
    steps = defaultdict(set)
    for x in load_jsonl(copy / "outgoing.jsonl"):
        steps[x["request_id"]].add(x["step"])
    after = {x["request_id"]: x for x in json.loads((copy / "leakage.json").read_text())}
    thr, min_len = policy.guard.get("fuzzy_threshold", 90), policy.guard.get("min_fuzzy_len", 4)

    c = Counter()
    checks, missed_reasons = Counter(), Counter()
    for i, r in res.items():
        ex = examples[i]
        before = leak_m.request_leakage(i, ex.private_spans, [MASK_INSTRUCTION + r["notes"]["masked_query"]],
                                        thr, min_len, query=ex.query)
        aft = after[i]
        first = log[i][0] if log[i] else None
        if first is None:
            outcome = "no decision"
        elif first["allow"]:
            outcome = "as labeled"
        elif "cloud_r1" in r["notes"].get("remasked", []) and len(log[i]) > 1 and log[i][1]["allow"]:
            outcome = "after hiding"
        else:
            outcome = "not sent"
        expected = {"as labeled": "cloud_r1", "after hiding": "cloud_r1_remasked"}.get(outcome)
        sent = {s for s in steps[i] if s.startswith("cloud_r1")}
        c["step_mismatch"] += (sent != ({expected} if expected else set()))
        c[outcome] += 1
        if first is not None and not first["allow"]:
            c["blocked"] += 1
            for name in {check_of(x) for x in first["reasons"]}:
                checks[name] += 1
        local_b = [s for s in before.spans if s.label == "local-only"]
        local_a = [s for s in aft["spans"] if s["label"] == "local-only"]
        hb, ha = before.hard_violation, aft["hard_violation"]
        c["hard_before"] += hb
        c["hard_after"] += ha
        c["hard_strict_before"] += before.hard_violation_strict
        c["hard_strict_after"] += aft["hard_violation_strict"]
        c["local_n"] += len(local_b)
        c["local_leaked_before"] += sum(s.leaked for s in local_b)
        c["local_leaked_after"] += sum(s["leaked"] for s in local_a)
        c["stopped"] += hb and not ha
        c["stopped_by_hiding"] += hb and not ha and outcome == "after hiding"
        c["stopped_by_not_sending"] += hb and not ha and outcome == "not sent"
        c["added"] += ha and not hb
        c["inconsistent"] += outcome == "as labeled" and hb != ha
        if outcome != "as labeled" and first is not None and not hb:
            c["blocked_clean"] += 1
            c["blocked_clean_not_sent"] += outcome == "not sent"
        if ha:
            missed_reasons[",".join(sorted({s["category"] for s in local_a if s["leaked"]}))] += 1
    return {"requests": len(res), "run": orig.name, **c, "checks": dict(checks),
            "missed_categories": dict(missed_reasons.most_common())}


def labeler(cur: dict, examples: dict, policy: Policy) -> dict:
    res = load_results(cur["open_run"].removesuffix("-no_gap_loop"))
    thr, min_len = policy.guard.get("fuzzy_threshold", 90), policy.guard.get("min_fuzzy_len", 4)
    spans = defaultdict(lambda: [0, 0])
    local_cat = defaultdict(lambda: [0, 0])
    words = Counter()
    unaligned = 0
    for i, r in res.items():
        ex, masked = examples[i], r["notes"]["masked_query"]
        lk = leak_m.request_leakage(i, ex.private_spans, [masked], thr, min_len, query=ex.query)
        for s in lk.spans:
            spans[s.label][0] += 1
            spans[s.label][1] += not s.leaked_strict
            if s.label == "local-only":
                local_cat[s.category][0] += 1
                local_cat[s.category][1] += not s.leaked_strict
        stretches = masked_stretches(ex.query, masked)
        if stretches is None:
            unaligned += 1
            continue
        gold = defaultdict(list)
        for s in ex.private_spans:
            gold[s.label.value] += occurrences(ex.query, s.text)
        private = gold["local-only"] + gold["abstractable"]
        for m in WORD.finditer(ex.query):
            w = m.span()
            words["all"] += 1
            hidden = overlaps(w, stretches)
            in_private = overlaps(w, private)
            words["private"] += in_private
            if hidden:
                words["hidden"] += 1
                words["hidden_private"] += in_private
                words["hidden_public"] += (not in_private) and overlaps(w, gold["public"])
            elif in_private:
                words["missed_private"] += 1
    by_label = {k: {"spans": n, "hidden": h, "recall": ratio(h, n)} for k, (n, h) in spans.items()}
    cats = {k: {"spans": n, "hidden": h, "recall": ratio(h, n)} for k, (n, h) in
            sorted(local_cat.items(), key=lambda kv: -kv[1][0])}
    h = words["hidden"]
    return {"requests": len(res), "unaligned": unaligned, "by_label": by_label, "local_only_by_category": cats,
            "words": dict(words), "word_precision": ratio(words["hidden_private"], h),
            "word_public_share": ratio(words["hidden_public"], h),
            "word_unlabeled_share": ratio(h - words["hidden_private"] - words["hidden_public"], h),
            "word_recall": ratio(words["hidden_private"], words["private"]),
            "words_hidden_share": ratio(h, words["all"])}


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def report(out: dict) -> str:
    L = ["# The method's parts on PUPA (stored runs, no model calls)", "",
         "Written by scripts/pupa_components.py. Shares are in %. Each split's own requests; test is the clean one.", ""]
    for split, models in out.items():
        L += [f"## {split}", "", "### Confidence gate (dev threshold, best quality)", "",
              "Sent / at least as good / hard violations. Random sends the same share at random (expected value); "
              "the oracle sends exactly the requests where the cloud's answer is better.", "",
              "| Edge model | Gate closed | Gate (dev t) | Random | Oracle | Gate open | Gate − open, quality [95%] | "
              "Gate − open, hard [95%] | Equal quality at | AUROC cloud better | AUROC edge good | ECE |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, d in models.items():
            g = d["gate"]
            f3 = lambda p: f"{pct(p['sent'], 0)} / {pct(p['quality'])} / {pct(p['hard'])}"
            dm = g["gate_minus_send_all"]
            eq = g["equal_quality"]
            L.append(f"| {MODELS[m]} | {f3(g['send_none'])} | {f3(g['gate'])} | {f3(g['random'])} | {f3(g['oracle'])} | "
                     f"{f3(g['send_all'])} | {pct(dm['quality'])} [{pct(dm['quality_ci'][0])}, {pct(dm['quality_ci'][1])}] | "
                     f"{pct(dm['hard'])} [{pct(dm['hard_ci'][0])}, {pct(dm['hard_ci'][1])}] | "
                     f"{pct(eq['sent'], 0) if eq else '–'} sent | {f2(g['auroc_cloud_better'])} | "
                     f"{f2(g['auroc_edge_good'])} | {f2(g['ece_edge_good'])} |")
        L += ["", "### Guard (gate open: every request tries the cloud once)", "",
              "| Edge model | Requests | First message blocked | By: known value / detector / request check / leak check | "
              "Sent after hiding | Not sent | Hard violations, labeler alone → after guard | Local-only items leaked | "
              "Blocked without a gold leak (of them not sent) | Step mismatches |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m, d in models.items():
            g, n = d["guard"], d["guard"]["requests"]
            ch = " / ".join(str(g["checks"].get(k, 0)) for k in CHECKS[:4])
            L.append(f"| {MODELS[m]} | {n} | {g.get('blocked', 0)} ({pct(ratio(g.get('blocked', 0), n), 0)}) | {ch} | "
                     f"{g.get('after hiding', 0)} | {g.get('not sent', 0)} | {g.get('hard_before', 0)} → "
                     f"{g.get('hard_after', 0)} ({pct(ratio(g.get('hard_before', 0), n))} → "
                     f"{pct(ratio(g.get('hard_after', 0), n))}) | {g.get('local_leaked_before', 0)} → "
                     f"{g.get('local_leaked_after', 0)} of {g.get('local_n', 0)} | {g.get('blocked_clean', 0)} "
                     f"({g.get('blocked_clean_not_sent', 0)}) | {g.get('step_mismatch', 0)} |")
        L += ["", "Leaks left after the guard, by the categories leaked (requests):", ""]
        L += [f"- {MODELS[m]}: " + (", ".join(f"{k} {v}" for k, v in d["guard"]["missed_categories"].items()) or "none")
              + (f"; added by the guard {d['guard']['added']}, inconsistent {d['guard']['inconsistent']}"
                 if d["guard"].get("added") or d["guard"].get("inconsistent") else "") for m, d in models.items()]
        L += ["", "### Labeler", "",
              "Recall: share of gold spans hidden (plain matching). Words: of the words hidden, the share in a gold "
              "local-only or abstractable value, in a public one, or in none.", "",
              "| Edge model | Local-only recall | Abstractable recall | Public hidden | Word precision | Hidden words: "
              "public / unlabeled | Word recall | Words hidden | Unaligned |", "|---|---|---|---|---|---|---|---|---|"]
        for m, d in models.items():
            b, lab = d["labeler"]["by_label"], d["labeler"]
            rec = lambda k: f"{pct(b[k]['recall'])} ({b[k]['hidden']}/{b[k]['spans']})" if k in b else "–"
            L.append(f"| {MODELS[m]} | {rec('local-only')} | {rec('abstractable')} | {rec('public')} | "
                     f"{pct(lab['word_precision'])} | {pct(lab['word_public_share'])} / {pct(lab['word_unlabeled_share'])} | "
                     f"{pct(lab['word_recall'])} | {pct(lab['words_hidden_share'])} | {lab['unaligned']} |")
        L += ["", "Local-only recall by category (spans hidden / spans):", ""]
        first = next(iter(models.values()))["labeler"]["local_only_by_category"]
        cats = [k for k, v in first.items() if v["spans"] >= 5]
        L += ["| Edge model | " + " | ".join(cats) + " | other categories |", "|---" * (len(cats) + 2) + "|"]
        for m, d in models.items():
            cc = d["labeler"]["local_only_by_category"]
            rest = [v for k, v in cc.items() if k not in cats]
            L.append(f"| {MODELS[m]} | " + " | ".join(f"{cc[k]['hidden']}/{cc[k]['spans']}" if k in cc else "–"
                                                      for k in cats)
                     + f" | {sum(v['hidden'] for v in rest)}/{sum(v['spans'] for v in rest)} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    policy = Policy.load()
    rng = np.random.default_rng(0)
    datasets: dict[str, dict] = {}
    out: dict[str, dict] = {}
    for split in args.splits:
        out[split] = {}
        for m in args.models:
            cur = json.loads((ROOT / SPLITS[split] / m / "curves.json").read_text())
            data = yaml.safe_load((ROOT / "runs" / cur["open_run"] / "config.yaml").read_text())["data"]["name"]
            if data not in datasets:
                datasets[data] = {e.id: e for e in read_examples(data)}
            threshold = json.loads((ROOT / DEV.format(m)).read_text())["best_quality"]["threshold"]
            out[split][m] = {"data": data, "open_run": cur["open_run"], "closed_run": cur["closed_run"],
                             "gate": gate(cur, threshold, rng), "guard": guard(cur, datasets[data], policy),
                             "labeler": labeler(cur, datasets[data], policy)}
            print(split, m, "done", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "components.json").write_text(json.dumps(out, indent=1))
    md = report(out)
    (args.out / "components.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
