#!/usr/bin/env python
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from gapdeleg.build.io import read_examples
from gapdeleg.metrics import leakage as leak_m
from gapdeleg.privacy.policy import Policy

C1 = "cloud_r1"


def ablate(task: Path, out: Path, examples: dict) -> dict:
    table = json.loads((task / "table.json").read_text())
    row = table["rows"][0]
    run = ROOT / "runs" / row["run"]
    new = ROOT / "runs" / f"{run.name}-no_gap_loop"
    new.mkdir(exist_ok=True)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    shutil.copy(run / "config.yaml", new / "config.yaml")
    shutil.copy(run / "metadata.json", new / "metadata.json")

    sent = [x for x in map(json.loads, (run / "outgoing.jsonl").read_text().splitlines()) if x["step"].startswith(C1)]
    (new / "outgoing.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sent))
    by_id: dict[str, list[dict]] = {}
    for x in sent:
        by_id.setdefault(x["request_id"], []).append(x)

    recs, changed = [], 0
    for r in map(json.loads, (run / "results.jsonl").read_text().splitlines()):
        before = r["notes"].get("answer_before_gaps", r["answer"])
        if before != r["answer"]:
            changed += 1
            r.pop("judge", None)
            r.pop("judge_error", None)
        mine = by_id.get(r["id"], [])
        r = dict(r, answer=before, cost=dict(r["cost"], cloud_calls=len(mine),
                                              cloud_cached_calls=sum(bool(x.get("cached")) for x in mine),
                                              cloud_in_tokens=sum(x["usage"]["input_tokens"] for x in mine),
                                              cloud_out_tokens=sum(x["usage"]["output_tokens"] for x in mine)))
        recs.append(r)
    (new / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))

    policy = Policy.load(cfg["policy"]) if cfg.get("policy") else Policy.load()
    texts = {i: [m["content"] for x in xs for m in x["messages"] if m["role"] != "assistant"] for i, xs in by_id.items()}
    leaks = [leak_m.request_leakage(r["id"], examples[r["id"]].private_spans, texts.get(r["id"], []),
                                    policy.guard.get("fuzzy_threshold", 90), policy.guard.get("min_fuzzy_len", 4),
                                    query=examples[r["id"]].query) for r in recs]
    (new / "leakage.json").write_text(json.dumps([lk.to_dict() for lk in leaks], indent=1, ensure_ascii=False))
    summary = json.loads((run / "summary.json").read_text())
    summary.pop("quality", None)
    summary.update(run_id=new.name, leakage=leak_m.aggregate(leaks), ablation="no gap loop", ablated_from=run.name)
    (new / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    hard = summary["leakage"]["hard_violation_rate"]
    row = dict(row, system="gap_aware (no gap loop)", run=new.name, at_least_as_good=None, hard_violation=hard,
               cloud_calls=sum(r["cost"]["cloud_calls"] for r in recs) / len(recs), cloud_spend_usd=None)
    (out / task.name).mkdir(parents=True, exist_ok=True)
    (out / task.name / "table.json").write_text(json.dumps(dict(table, rows=[row]), indent=1))
    return {"task": task.name, "run": new.name, "answers_changed_by_gap_loop": changed, "hard_violation": hard}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", help="run_method.py output directories")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    datasets: dict[str, dict] = {}
    for task in map(Path, args.tasks):
        run = ROOT / "runs" / json.loads((task / "table.json").read_text())["rows"][0]["run"]
        name = yaml.safe_load((run / "config.yaml").read_text())["data"]["name"]
        if name not in datasets:
            datasets[name] = {e.id: e for e in read_examples(name)}
        print(json.dumps(ablate(task, Path(args.out), datasets[name])))


if __name__ == "__main__":
    main()
