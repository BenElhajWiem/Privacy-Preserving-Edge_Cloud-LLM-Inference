#!/usr/bin/env python
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["qwen3-1.7b", "qwen3-4b-q4_K_M", "qwen3-4b-instruct-2507-q4_K_M", "llama3.2-3b", "gemma3-4b-it-q4_K_M"]


def first_row(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d["rows"][0] if "rows" in d else d


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, help="default <runs>/famb-test/attacker")
    args = ap.parse_args()
    t = args.runs / "famb-test"
    for ds in ("privfin", "privmed"):
        cloud = first_row(t / "cloud-only" / ds / "table.json")
        rows = []
        for m in MODELS:
            method = first_row(t / "method" / "rows" / f"{ds}_test__{m}.json")
            edge = method["edge"]
            for system, src in (("edge_only", t / "edge-only" / f"{ds}-{m}" / "table.json"),
                                ("redact_send", t / "redact-send" / f"{ds}-{m}" / "table.json")):
                r = first_row(src)
                rows.append({"edge": edge, "system": system, "run": r["run"]})
            rows.append({"edge": edge, "system": "method", "run": method["run"]})
            rows.append({"edge": edge, "system": "cloud_full", "run": cloud["run"]})
        out = (args.out or t / "attacker") / ds
        out.mkdir(parents=True, exist_ok=True)
        (out / "table.json").write_text(json.dumps({"data": f"{ds}_test", "rows": rows}, indent=1))
        print(out / "table.json", len(rows), "rows")


if __name__ == "__main__":
    main()
