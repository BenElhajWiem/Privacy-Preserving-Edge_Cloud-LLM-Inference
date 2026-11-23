#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GOOD, BAD = {"win", "tie"}, {"loss"}


def run_dir(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if (p / "results.jsonl").exists():
        return p
    table = json.loads((p / "table.json").read_text())
    rows = table["rows"] if isinstance(table, dict) and "rows" in table else table
    runs = [r["run"] for r in rows if isinstance(r, dict) and r.get("run")]
    if len(runs) != 1:
        sys.exit(f"{p}: expected one run in table.json, found {runs}")
    q = Path(runs[0])
    return q if q.is_absolute() else ROOT / (q if str(q).startswith("runs/") else Path("runs") / q)


def outcome(rec: dict):
    if not (rec.get("answer") or "").strip():
        return 0
    verdict = (rec.get("judge") or {}).get("verdict")
    return 1 if verdict in GOOD else 0 if verdict in BAD else None


def settings(d: Path) -> dict:
    meta = json.loads((d / "metadata.json").read_text())
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    edge = meta.get("models", {}).get("edge", {})
    run = cfg.get("gen", {}) if isinstance(cfg.get("gen"), dict) else {}
    return {"run": d.name, "started": meta.get("started"), "platform": meta.get("platform"),
            "model": edge.get("model"), "digest": (edge.get("digest") or "")[:12],
            "ollama": edge.get("ollama_version"), "think": edge.get("think"),
            "num_ctx": (cfg.get("edge") or {}).get("num_ctx"),
            "edge_max_tokens": run.get("edge_max_tokens", cfg.get("edge_max_tokens"))}


def compare(label: str, edge_path: str, method_path: str) -> dict:
    de, dm = run_dir(edge_path), run_dir(method_path)
    e = {r["id"]: r for r in map(json.loads, (de / "results.jsonl").read_text().splitlines())}
    m = {r["id"]: r for r in map(json.loads, (dm / "results.jsonl").read_text().splitlines())}
    ids = sorted(set(e) & set(m))
    same = [i for i in ids if (e[i].get("answer") or "").strip() == (m[i].get("answer") or "").strip()]
    judged = [i for i in ids if outcome(e[i]) is not None and outcome(m[i]) is not None]
    agree = [i for i in judged if outcome(e[i]) == outcome(m[i])]
    same_j = [i for i in judged if i in set(same)]
    diff_j = [i for i in judged if i not in set(same)]
    se, sm = settings(de), settings(dm)
    return {
        "label": label, "n": len(ids), "identical": len(same), "identical_pct": round(100 * len(same) / max(len(ids), 1), 1),
        "judged_both": len(judged),
        "edge_only_rate": round(100 * sum(outcome(e[i]) for i in judged) / max(len(judged), 1), 1),
        "draft_rate": round(100 * sum(outcome(m[i]) for i in judged) / max(len(judged), 1), 1),
        "edge_only_wins": sum(outcome(e[i]) for i in judged), "draft_wins": sum(outcome(m[i]) for i in judged),
        "outcome_agree": len(agree),
        "agree_when_identical": f"{sum(outcome(e[i]) == outcome(m[i]) for i in same_j)}/{len(same_j)}",
        "agree_when_different": f"{sum(outcome(e[i]) == outcome(m[i]) for i in diff_j)}/{len(diff_j)}",
        "only_edge_good": sum(outcome(e[i]) == 1 and outcome(m[i]) == 0 for i in judged),
        "only_draft_good": sum(outcome(e[i]) == 0 and outcome(m[i]) == 1 for i in judged),
        "same_settings": {k: se[k] == sm[k] for k in ("model", "digest", "ollama", "think", "num_ctx", "edge_max_tokens")},
        "same_platform": se["platform"] == sm["platform"],
        "edge_only_run": se, "method_run": sm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, metavar="LABEL=EDGE,METHOD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for p in args.pair:
        label, rest = p.split("=", 1)
        edge, method = rest.split(",", 1)
        rows.append(compare(label, edge, method))
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "noise.json").write_text(json.dumps(rows, indent=1))
    lines = ["| Pair | n | Identical answers | Edge-only ≥ ref | Draft ≥ ref | Outcomes agree | "
             "Agree, identical / different answers | Only edge-only good / only draft good | Same settings | Same platform | Dates |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['n']} | {r['identical']} ({r['identical_pct']}%) | {r['edge_only_wins']} ({r['edge_only_rate']}%) "
                     f"| {r['draft_wins']} ({r['draft_rate']}%) | {r['outcome_agree']}/{r['judged_both']} "
                     f"| {r['agree_when_identical']} / {r['agree_when_different']} | {r['only_edge_good']} / {r['only_draft_good']} "
                     f"| {'yes' if all(r['same_settings'].values()) else r['same_settings']} | {'yes' if r['same_platform'] else 'no'} "
                     f"| {(r['edge_only_run']['started'] or '')[:10]} / {(r['method_run']['started'] or '')[:10]} |")
    (out / "noise.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in rows:
        if not r["same_platform"]:
            print(f"{r['label']}: platform {r['edge_only_run']['platform']} vs {r['method_run']['platform']}")


if __name__ == "__main__":
    main()
