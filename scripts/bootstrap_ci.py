#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.build.io import id_order
from gapdeleg.metrics.bootstrap import confidence_intervals, paired_test, request_outcomes, resamples

COLUMNS = [("at_least_as_good", "≥ reference"), ("hard_violation", "Hard violations"),
           ("span_leak_local_only", "Local-only spans leaked"), ("abstraction_compliance", "Abstraction compliance"),
           ("attacker_reconstruction", "Attacker")]


def systems_by_edge(table: Path, extra: list[str]) -> dict[str, dict[str, list[dict]]]:
    groups: dict[str, dict[str, list[dict]]] = {}
    for r in json.loads(table.read_text())["rows"]:
        if "run" not in r:
            continue
        label = ("prism (mean of seeds)" if r["system"] == "prism" else
                 f"{r['system']} ({r['threshold_rule'].replace('_', ' ')})" if "threshold_rule" in r else r["system"])
        if r.get("tag"):
            label += f" [{r['tag']}]"
        groups.setdefault(r["edge"], {}).setdefault(label, []).append(request_outcomes(ROOT / "runs" / r["run"]))
    for item in extra:
        label, path = item.split("=", 1)
        d = json.loads(Path(path).read_text())
        groups.setdefault(d["edge"], {})[label] = [d["outcomes"]]
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", type=Path)
    ap.add_argument("--against", help="system for paired tests, e.g. redact_send")
    ap.add_argument("--extra", nargs="*", default=[], metavar="LABEL=FILE")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def pct(x):
        return f"{100 * x:.1f}"

    def pval(p):
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"
    out, lines = {}, [f"# 95% bootstrap confidence intervals over requests ({args.resamples} resamples, seed {args.seed})", ""]
    for edge, systems in sorted(systems_by_edge(args.table, args.extra).items()):
        ids = sorted(set.intersection(*(set(o) for runs in systems.values() for o in runs)), key=id_order)
        idx = resamples(len(ids), args.resamples, args.seed)
        ci = {name: confidence_intervals(runs, ids, idx) for name, runs in systems.items()}
        out[edge] = {"requests": len(ids), "ci": ci}
        lines += [f"## Edge {edge} ({len(ids)} requests)", "",
                  "| System | " + " | ".join(f"{h} (%)" for _, h in COLUMNS) + " | Cloud tokens in/out |",
                  "|---" * (len(COLUMNS) + 2) + "|"]
        for name, c in ci.items():
            cells = [f"{pct(c[m]['value'])} [{pct(c[m]['low'])}, {pct(c[m]['high'])}]" for m, _ in COLUMNS]
            lines.append(f"| {name} | " + " | ".join(cells) +
                         f" | {c['cloud_in_tokens']['value']:.0f} / {c['cloud_out_tokens']['value']:.0f} |")
        lines.append("")
        if args.against and args.against in systems:
            base = systems[args.against]
            tests = {name: paired_test(runs, base, ids, idx) for name, runs in systems.items() if name != args.against}
            out[edge]["paired_against"] = {"system": args.against, "tests": tests}
            lines += [f"Paired against {args.against}: difference in points [95% interval], two-sided p.", "",
                      "| System | " + " | ".join(f"Δ {h}" for _, h in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
            for name, t in tests.items():
                lines.append(f"| {name} | " + " | ".join(
                    f"{100 * t[m]['diff']:+.1f} [{100 * t[m]['low']:+.1f}, {100 * t[m]['high']:+.1f}], {pval(t[m]['p'])}"
                    for m, _ in COLUMNS) + " |")
            lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ci.json").write_text(json.dumps(out, indent=1))
    (args.out / "ci.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
