#!/usr/bin/env python
import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.metrics.bootstrap import point_values, request_outcomes

COLUMNS = [("at_least_as_good", "≥ reference"), ("hard_violation", "Hard violations"),
           ("span_leak_local_only", "Local-only spans leaked"), ("attacker_reconstruction", "Attacker"),
           ("cloud_out_tokens", "Cloud output tokens")]


def sent_messages(run: Path) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    p = run / "outgoing.jsonl"
    for o in map(json.loads, p.read_text().splitlines() if p.exists() else []):
        out.setdefault(o["request_id"], []).append(json.dumps(o["messages"], sort_keys=True, ensure_ascii=False))
    return {k: "\n".join(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--extra", nargs="*", default=[], metavar="LABEL=GLOB")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    systems: dict[str, list[dict]] = {}
    for d in args.dirs:
        row = json.loads((Path(d) / "table.json").read_text())["rows"][0]
        run = ROOT / "runs" / row["run"]
        systems.setdefault(row["system"], []).append(
            {"tag": row.get("tag"), "values": point_values(list(request_outcomes(run).values())),
             "sent": sent_messages(run)})
    for item in args.extra:
        label, pattern = item.split("=", 1)
        for f in sorted(glob.glob(pattern)):
            d = json.loads(Path(f).read_text())
            systems.setdefault(label, []).append({"tag": f, "values": point_values(list(d["outcomes"].values()))})

    out, lines = {}, ["# Cloud sampling variance over uncached repeats", "",
                      "Mean, [min–max] and standard deviation over the repeats, in points (tokens for cloud output).", "",
                      "| System | Repeats | " + " | ".join(h for _, h in COLUMNS) + " | Same cloud messages in every repeat (%) |",
                      "|---" * (len(COLUMNS) + 3) + "|"]
    for name, reps in systems.items():
        stats = {}
        for m, _ in COLUMNS:
            vals = [r["values"][m] for r in reps if r["values"][m] is not None]
            stats[m] = {"values": vals, "mean": statistics.mean(vals), "min": min(vals), "max": max(vals),
                        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        same = None
        if all("sent" in r for r in reps):
            ids = set().union(*(r["sent"] for r in reps))
            same = sum(len({r["sent"].get(i) for r in reps}) == 1 for i in ids) / len(ids) if ids else None
        out[name] = {"repeats": [r["tag"] for r in reps], "metrics": stats, "same_cloud_messages": same}

        def cell(m):
            k = 1 if m == "cloud_out_tokens" else 100
            st = stats[m]
            return f"{k * st['mean']:.1f} [{k * st['min']:.1f}–{k * st['max']:.1f}], sd {k * st['sd']:.1f}"
        lines.append(f"| {name} | {len(reps)} | " + " | ".join(cell(m) for m, _ in COLUMNS) +
                     f" | {'–' if same is None else f'{100 * same:.1f}'} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "variance.json").write_text(json.dumps(out, indent=1))
    (args.out / "variance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
