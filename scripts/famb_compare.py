#!/usr/bin/env python
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from famb_stats import paired
from famb_common import EDGES, NAMES
from famb_baselines import CAPS

from gapdeleg.data import load
from gapdeleg.experiment import load_outgoing
from gapdeleg.metrics.leakage import request_leakage
from gapdeleg.metrics.option_list import leakage_without_options, masks_an_option

TEST = ROOT / "runs" / "famb-test"
CAPS_USD = {"stage 1": 9.0, "stage 2": 16.0}
SETS = {"privfin_test": "PrivFin", "privmed_test": "PrivMed"}
COUNTED = {"privfin_test": ["qwen3:1.7b", "llama3.2:3b", "gemma3:4b-it-q4_K_M"], "privmed_test": EDGES}
COUNTS = {"hard_measured": "as measured", "hard_rule": "option-list rule"}
HARD = {"hard_measured": 1, "hard_rule": 2}
BASELINES = {"edge_only": "edge-only", "redact_send": "redact-and-send", "cloud_full": "cloud-only"}


def edge_slug(edge: str) -> str:
    return edge.replace(":", "-")


def outcomes(run: Path, examples: list) -> tuple[dict, int]:
    recs = {r["id"]: r for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())}
    stored = {x["request_id"]: x["hard_violation"] for x in json.loads((run / "leakage.json").read_text())}
    sent = load_outgoing(run / "outgoing.jsonl")
    out, mismatch = {}, 0
    for ex in examples:
        messages = sent.get(ex.id, [])
        measured = request_leakage(ex.id, ex.private_spans, messages, 90, 4, query=ex.query).hard_violation
        mismatch += ex.id in stored and bool(stored[ex.id]) != bool(measured)
        out[ex.id] = (int(bool(recs.get(ex.id, {}).get("exact_match"))), int(measured),
                      int(leakage_without_options(ex, messages).hard_violation))
    return out, mismatch


def best_baseline(baselines: dict[str, dict], count: str) -> str:
    clean = {k: b for k, b in baselines.items() if b[count] == 0}
    return max(clean, key=lambda k: clean[k]["exact"])


def meets(cell: dict, count: str) -> bool:
    test = cell.get("vs_best", {}).get(count)
    return bool(test) and cell[count] == 0 and test["diff"] > 0 and test["p"] < 0.05


def verdict(cells: list[dict], count: str) -> dict:
    counted = [c for c in cells if c["counted"]]
    passed = [c for c in counted if meets(c, count)]
    need = max(2, math.ceil(len(counted) / 2))
    if any(c.get("missing") for c in counted):
        state = "incomplete"
    else:
        state = ("no_go" if len(passed) < need else
                 "go" if all(c.get("stage_2_done") for c in counted) else "needs_stage_2")
    return {"counted": len(counted), "need": need, "passed": len(passed), "state": state,
            "pairs": [[c["data"], c["edge"]] for c in passed]}


def masked_options(run: Path, examples: list) -> int:
    sent = load_outgoing(run / "outgoing.jsonl")
    return sum(any(masks_an_option(m, ex.meta.get("options") or []) for m in sent.get(ex.id, [])) for ex in examples)


def verdict_if_masked_counted(cells: list[dict]) -> dict:
    return verdict([{**c, "hard_rule": c.get("hard_rule", 0) + c.get("masked_options", 0)} for c in cells], "hard_rule")


def digest(run: Path) -> str | None:
    return json.loads((run / "metadata.json").read_text())["models"]["edge"].get("digest")


def baseline_result(table: Path, examples: list) -> dict | None:
    if not table.exists():
        return None
    row = json.loads(table.read_text())["rows"][0]
    run = ROOT / "runs" / row["run"]
    o, mismatch = outcomes(run, examples)
    return {"_o": o, "exact": sum(v[0] for v in o.values()) / len(o), "hard_measured": sum(v[1] for v in o.values()),
            "hard_rule": sum(v[2] for v in o.values()), **strict_counts(run, examples), "metric_mismatch": mismatch,
            "requests_run": row["n"],
            "errors": row["errors"], "empty_answers": row["empty_answers"],
            "cloud_spend_usd": row.get("cloud_spend_usd") or 0,
            "stopped_at_budget": json.loads((run / "summary.json").read_text()).get("stopped_at_budget"),
            "digest": digest(run), "run": row["run"]}


def cell_for(data: str, edge: str, examples: list, cloud_only: dict | None = None) -> dict:
    cell = {"data": data, "edge": edge, "counted": edge in COUNTED[data], "n": len(examples)}
    row_path = TEST / "method" / "rows" / f"{data}__{edge_slug(edge)}.json"
    family = data.split("_")[0]
    table = TEST / "edge-only" / f"{family}-{edge_slug(edge)}" / "table.json"
    if not row_path.exists() or not table.exists():
        return {**cell, "missing": True}
    row = json.loads(row_path.read_text())
    base = json.loads(table.read_text())["rows"][0]
    method_run, base_run = ROOT / "runs" / row["run"], ROOT / "runs" / base["run"]
    method, method_mismatch = outcomes(method_run, examples)
    edge_only, base_mismatch = outcomes(base_run, examples)
    recs = [json.loads(x) for x in (method_run / "results.jsonl").read_text().splitlines()]
    baselines = {"edge_only": {"_o": edge_only, "exact": sum(v[0] for v in edge_only.values()) / len(edge_only),
                               "hard_measured": sum(v[1] for v in edge_only.values()),
                               "hard_rule": sum(v[2] for v in edge_only.values())}}
    redact = baseline_result(TEST / "redact-send" / f"{family}-{edge_slug(edge)}" / "table.json", examples)
    for name, result in (("redact_send", redact), ("cloud_full", cloud_only)):
        if result and not result["stopped_at_budget"]:
            baselines[name] = result
    best = {count: best_baseline(baselines, count) for count in COUNTS}
    cell.update({"stage_2_done": len(baselines) == 3,
                 "best": best, "vs_best": {count: {"baseline": b, **paired(method, baselines[b]["_o"])}
                                           for count, b in best.items()},
                 "redact_send": redact and {k: v for k, v in redact.items() if k != "_o"},
                 "same_weights_redact": redact is None or redact["digest"] == digest(method_run)})
    if redact:
        bound = zero_violation_bound(edge_only, redact["_o"])
        cell.update({"zero_violation_bound": sum(v[0] for v in bound.values()) / len(bound),
                     "vs_bound": paired(method, bound)})
    return {**cell, "edge_only": sum(v[0] for v in edge_only.values()) / len(edge_only),
            "method": sum(v[0] for v in method.values()) / len(method),
            "vs_edge_only": paired(method, edge_only),
            "hard_measured": sum(v[1] for v in method.values()), "hard_rule": sum(v[2] for v in method.values()),
            **strict_counts(method_run, examples), "masked_options": masked_options(method_run, examples),
            "edge_only_hard_measured": sum(v[1] for v in edge_only.values()),
            "edge_only_hard_rule": sum(v[2] for v in edge_only.values()),
            "metric_mismatch": method_mismatch + base_mismatch, "errors": row["errors"],
            "empty_answers": row["empty_answers"], "edge_only_errors": base["errors"],
            "edge_only_empty": base["empty_answers"], "stopped_at_budget": row.get("stopped_at_budget"),
            "requests_run": row["n"], "budget_usd": row["budget_usd"], "cloud_spend_usd": row.get("cloud_spend_usd") or 0,
            "cloud_calls": row["cloud_calls"], "cloud_fresh_calls": row["cloud_fresh_calls"],
            "routes": dict(Counter(r["notes"].get("route") for r in recs)),
            "same_weights": digest(method_run) == digest(base_run), "run": row["run"], "edge_only_run": base["run"]}


def strict_counts(run: Path, examples: list) -> dict:
    sent = load_outgoing(run / "outgoing.jsonl")
    return {"strict_measured": sum(request_leakage(ex.id, ex.private_spans, sent.get(ex.id, []), 90, 4,
                                                   query=ex.query).hard_violation_strict for ex in examples),
            "strict_rule": sum(leakage_without_options(ex, sent.get(ex.id, [])).hard_violation_strict for ex in examples)}


def verdict_if_strict(cells: list[dict], count: str) -> dict:
    return verdict([{**c, "hard_rule": c.get(count, 0)} for c in cells], "hard_rule")


def zero_violation_bound(edge_only: dict, redact: dict) -> dict:
    return {i: (max(e[0], 0 if redact[i][2] else redact[i][0]),) for i, e in edge_only.items()}


def markdown(report: dict) -> str:
    def pct(x):
        return f"{100 * x:.1f}"

    def pval(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    out = ["# Family B test sets: the method against the baselines", "",
           "Exact match in %. Hard violations are requests, as measured / under the option-list rule. Paired bootstrap, "
           "10,000 resamples, two-sided, against the best zero-violation baseline under the option-list rule. \"Meets\" "
           "uses the option-list rule; \"as measured\" follows it where different.", ""]
    for data, name in SETS.items():
        out += [f"## {name} ({report['sets'][data][0]['n']} requests)", ""]
        cloud = report["cloud_only"].get(data)
        if cloud:
            out += [f"Cloud-only: {pct(cloud['exact'])}, hard violations {cloud['hard_measured']} / {cloud['hard_rule']}.", ""]
        out += ["| Edge model | Edge-only | Redact-and-send | Its hard violations | Method | Its hard violations | Best baseline | "
                "Difference (95% CI) | p | Vs edge-only | Counted | Meets |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in report["sets"][data]:
            if c.get("missing"):
                out.append(f"| {NAMES[c['edge']]} | not run yet | | | | | | | | | {'yes' if c['counted'] else 'no'} | |")
                continue
            t, e, r = c["vs_best"]["hard_rule"], c["vs_edge_only"], c["redact_send"]
            mark = "yes" if meets(c, "hard_rule") else "no"
            if meets(c, "hard_measured") != meets(c, "hard_rule"):
                mark += " (as measured: " + ("yes" if meets(c, "hard_measured") else "no") + ")"
            redact = f"{pct(r['exact'])} | {r['hard_measured']} / {r['hard_rule']}" if r else "not run yet | "
            out.append(f"| {NAMES[c['edge']]} | {pct(c['edge_only'])} | {redact} | {pct(c['method'])} | "
                       f"{c['hard_measured']} / {c['hard_rule']} | {BASELINES[t['baseline']]} | {100 * t['diff']:+.1f} "
                       f"({100 * t['low']:+.1f} to {100 * t['high']:+.1f}) | {pval(t['p'])} | {100 * e['diff']:+.1f}, {pval(e['p'])} | "
                       f"{'yes' if c['counted'] else 'no'} | {mark if c['counted'] else '–'} |")
        out.append("")
    out += ["## Against the Go rule", ""]
    for count, label in COUNTS.items():
        v = report["verdict"][count]
        state = {"incomplete": "incomplete", "no_go": "No-go", "go": "Go",
                 "needs_stage_2": "not ruled out: a Go still needs stage 2 to run whole"}[v["state"]]
        out.append(f"- **{label}:** {v['passed']} of {v['counted']} pairs, needs {v['need']}: {state}.")
    v = report["verdict_if_masked_counted"]
    state = {"incomplete": "incomplete", "no_go": "No-go", "go": "Go",
             "needs_stage_2": "not ruled out: a Go still needs stage 2 to run whole"}[v["state"]]
    masked = ", ".join(f"{NAMES[c['edge']]} {c.get('masked_options', 0)}" for c in report["sets"]["privmed_test"])
    out += [f"- **if masked options counted as hard violations (not the rule):** {v['passed']} of {v['counted']} pairs, needs "
            f"{v['need']}: {state}.", "", f"Method requests that sent the option list with a word masked, PrivMed: {masked}."]
    out += ["", "Strict score (plain matching for every span, not the rule):"]
    for count, label in (("strict_rule", "option names removed"), ("strict_measured", "as measured")):
        v = report["verdict_if_strict"][count]
        out.append(f"- **{label}:** {v['passed']} of {v['counted']} pairs, needs {v['need']}: "
                   f"{'Go' if v['state'] == 'go' else 'No-go'}.")
    for data, name in SETS.items():
        cells = report["sets"][data]
        out.append(f"- **The method's requests, {name}, option names removed / as measured:** "
                   + ", ".join(f"{NAMES[c['edge']]} {c['strict_rule']} / {c['strict_measured']}" for c in cells) + ".")
    out += ["", "Upper limit for any cascade or AutoMix with zero hard violations (exact match in %, read off the edge-only and "
                "redact-and-send runs), and the method against it:"]
    for data, name in SETS.items():
        for c in report["sets"][data]:
            b = c["vs_bound"]
            out.append(f"- {name}, {NAMES[c['edge']]}: {pct(c['zero_violation_bound'])}; {100 * b['diff']:+.1f} "
                       f"({100 * b['low']:+.1f} to {100 * b['high']:+.1f}), {pval(b['p'])}.")
    out += ["", "## Checks: the method and edge-only", "",
            "| Set | Edge model | Requests run | Errors | Empty | Edge-only errors / empty | Edge-only hard | Stopped at cap | "
            "Fresh / all Claude calls | Spend ($, cap) | Metric reruns differing | Same weights | Routes |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for data, name in SETS.items():
        for c in report["sets"][data]:
            if not c.get("missing"):
                out.append(f"| {name} | {NAMES[c['edge']]} | {c['requests_run']} | {c['errors']} | {c['empty_answers']} | "
                           f"{c['edge_only_errors']} / {c['edge_only_empty']} | "
                           f"{c['edge_only_hard_measured']} / {c['edge_only_hard_rule']} | {c['stopped_at_budget'] or 'no'} | "
                           f"{c['cloud_fresh_calls']} / {c['cloud_calls']} | {c['cloud_spend_usd']:.2f}, {c['budget_usd']:.2f} | "
                           f"{c['metric_mismatch']} | {'yes' if c['same_weights'] else 'NO'} | {c['routes']} |")
    out += ["", "## Checks: stage 2", "",
            "| Set | Baseline | Edge model | Requests run | Errors | Empty | Stopped at cap | Spend ($, cap) | "
            "Metric reruns differing | Same weights as the method |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for data, name in SETS.items():
        rows = [("redact_send", c["edge"], c["redact_send"], c["same_weights_redact"])
                for c in report["sets"][data] if c.get("redact_send")]
        if report["cloud_only"].get(data):
            rows.append(("cloud_full", "–", report["cloud_only"][data], True))
        for system, edge, r, same in rows:
            out.append(f"| {name} | {BASELINES[system]} | {NAMES.get(edge, edge)} | {r['requests_run']} | {r['errors']} | "
                       f"{r['empty_answers']} | {r['stopped_at_budget'] or 'no'} | {r['cloud_spend_usd']:.2f}, "
                       f"{CAPS[system, data]:.2f} | {r['metric_mismatch']} | {'yes' if same else 'NO'} |")
    for r in report["stopped_runs"]:
        out += ["", f"Stopped at its cap and rerun: {r['data']} {NAMES[r['edge']]}, {r['n']} requests, "
                    f"${r['cloud_spend_usd']:.2f} ({r['run']})."]
    for r in report["failed_runs"]:
        out += ["", f"Credit ran out and rerun: redact-and-send {r['tag']} {NAMES[r['edge']]}, {r['errors']} of {r['n']} "
                    f"requests failed, ${r['cloud_spend_usd']:.2f} ({r['run']})."]
    for stage, spent in report["spend_usd"].items():
        out += ["", f"Claude spend in {stage}, stopped and failed runs included: ${spent:.2f} of the "
                    f"${CAPS_USD[stage]:.0f} approved."]
    return "\n".join(out) + "\n"


def main():
    report = {"sets": {}, "cloud_only": {}}
    stage_2 = 0.0
    for data in SETS:
        examples = load("processed", name=data)
        cloud = baseline_result(TEST / "cloud-only" / data.split("_")[0] / "table.json", examples)
        report["sets"][data] = [cell_for(data, edge, examples, cloud) for edge in EDGES]
        if cloud:
            report["cloud_only"][data] = {k: v for k, v in cloud.items() if k != "_o"}
            stage_2 += cloud["cloud_spend_usd"]
    cells = [c for group in report["sets"].values() for c in group]
    report["verdict"] = {count: verdict(cells, count) for count in COUNTS}
    report["verdict_if_masked_counted"] = verdict_if_masked_counted(cells)
    report["verdict_if_strict"] = {count: verdict_if_strict(cells, count) for count in ("strict_rule", "strict_measured")}
    stopped = [json.loads(p.read_text()) for p in sorted((TEST / "method" / "rows-stopped").glob("*.json"))]
    report["stopped_runs"] = [{k: r.get(k) for k in ("data", "edge", "n", "stopped_at_budget", "cloud_spend_usd", "run")}
                              for r in stopped]
    failed = [json.loads(p.read_text())["rows"][0] for p in sorted((TEST / "redact-send-failed").glob("*/table.json"))]
    report["failed_runs"] = [{k: r.get(k) for k in ("edge", "tag", "n", "errors", "cloud_spend_usd", "run")} for r in failed]
    stage_2 += sum((c.get("redact_send") or {}).get("cloud_spend_usd", 0) for c in cells)
    stage_2 += sum(r.get("cloud_spend_usd") or 0 for r in failed)
    report["spend_usd"] = {"stage 1": round(sum(c.get("cloud_spend_usd", 0) for c in cells)
                                            + sum(r.get("cloud_spend_usd") or 0 for r in stopped), 4),
                           "stage 2": round(stage_2, 4)}
    report["cap_usd"] = CAPS_USD
    text = markdown(report)
    (TEST / "table.json").write_text(json.dumps(report, indent=1))
    (TEST / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
