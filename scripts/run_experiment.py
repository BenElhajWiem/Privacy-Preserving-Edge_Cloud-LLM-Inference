#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env"))
from envfile import load_env

from gapdeleg.experiment import run


def apply_override(cfg: dict, expr: str) -> None:
    key, value = expr.split("=", 1)
    node = cfg
    *path, last = key.split(".")
    for p in path:
        node = node.setdefault(p, {})
    node[last] = yaml.safe_load(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. edge.model=qwen3:4b")
    args = ap.parse_args()
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.limit:
        cfg["data"]["limit"] = args.limit
    for s in args.set:
        apply_override(cfg, s)
    rd = run(cfg)
    summary = json.loads((rd.path / "summary.json").read_text())
    print(json.dumps({"run_dir": str(rd.path), "leakage": summary["leakage"],
                      "quality": summary.get("quality")}, indent=1))


if __name__ == "__main__":
    main()
