#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def load(run: str):
    p = Path(run)
    recs = {r["id"]: r for r in map(json.loads, (p / "results.jsonl").read_text().splitlines())}
    sent: dict[str, list[str]] = {}
    if (p / "outgoing.jsonl").exists():
        for o in map(json.loads, (p / "outgoing.jsonl").read_text().splitlines()):
            sent.setdefault(o["request_id"], []).append(json.dumps(o["messages"], sort_keys=True, ensure_ascii=False))
    summary = json.loads((p / "summary.json").read_text())
    meta = json.loads((p / "metadata.json").read_text())
    return recs, sent, summary, meta


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}" if d else "–"


def metric(s: dict) -> dict:
    return {"quality": s.get("quality", {}).get("at_least_as_good"),
            "hard": s["leakage"]["hard_violation_rate"],
            "local_only": s["leakage"]["per_label"].get("local-only", {}).get("rate")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="pairs of run directories: mac cluster [mac cluster ...]")
    ap.add_argument("--out")
    args = ap.parse_args()
    if len(args.runs) % 2:
        ap.error("give runs in pairs: <mac run> <cluster run>")

    lines = ["| System | Edge | Requests | Same answer (%) | Same cloud messages (%) | Same verdict (%) | "
             "≥ reference Mac / cluster (%) | Hard violations Mac / cluster (%) | Local-only leaked Mac / cluster (%) |",
             "|---|---|---|---|---|---|---|---|---|"]
    machines = set()
    for mac_run, cluster_run in zip(args.runs[::2], args.runs[1::2]):
        (ra, sa, ma, meta_a), (rb, sb, mb, meta_b) = load(mac_run), load(cluster_run)
        ids = sorted(set(ra) & set(rb))
        same_answer = sum(ra[i]["answer"].strip() == rb[i]["answer"].strip() for i in ids)
        sending = [i for i in ids if sa.get(i) or sb.get(i)]
        same_sent = sum(sa.get(i) == sb.get(i) for i in sending)
        judged = [i for i in ids if ra[i].get("judge") and rb[i].get("judge")]
        same_verdict = sum(ra[i]["judge"]["verdict"] == rb[i]["judge"]["verdict"] for i in judged)
        a, b = metric(ma), metric(mb)

        def pair(k):
            return " / ".join("–" if x[k] is None else f"{100 * x[k]:.1f}" for x in (a, b))
        edge = meta_a["models"]["edge"]
        lines.append(f"| {ma['system']} | {edge['model']} | {len(ids)} | {pct(same_answer, len(ids))} | "
                     f"{pct(same_sent, len(sending))} | {pct(same_verdict, len(judged))} | {pair('quality')} | "
                     f"{pair('hard')} | {pair('local_only')} |")
        for m in (meta_a, meta_b):
            e = m["models"]["edge"]
            machines.add(f"{m.get('platform', '?')}, Ollama {e.get('ollama_version')}, {e['model']} {e.get('digest', '')[:12]}")
    text = "\n".join(["Machines: " + "; ".join(sorted(machines)), ""] + lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
