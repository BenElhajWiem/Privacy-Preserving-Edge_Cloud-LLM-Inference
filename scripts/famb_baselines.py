#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import run_baselines

from gapdeleg import experiment

CAPS = {("redact_send", "privfin_test"): 1.50, ("redact_send", "privmed_test"): 1.30,
        ("cloud_full", "privfin_test"): 0.75, ("cloud_full", "privmed_test"): 1.25}
DEV_SETS = ("privfin_dev", "privmed_dev")
CHECK_CAP = 0.03
MAX_CHECK = 5


def cap_for(data: str, system: str, sample: int | None) -> float:
    if data in DEV_SETS:
        if system not in ("redact_send", "cloud_full") or not (sample and 0 < sample <= MAX_CHECK):
            sys.exit(f"a dev set is only a check: one of redact_send, cloud_full and --sample {MAX_CHECK} or less")
        return CHECK_CAP
    if (system, data) not in CAPS:
        sys.exit(f"no cap for {system} on {data}: one system per run, from {sorted(CAPS)}")
    if sample != 0:
        sys.exit("a test set runs whole: --sample 0")
    return CAPS[system, data]


def with_cap(run, cap: float):
    def capped(cfg, *args, **kw):
        return run({**cfg, "cloud_budget_usd": cap}, *args, **kw)
    return capped


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--sample", type=int)
    args, _ = ap.parse_known_args()
    experiment.run = with_cap(experiment.run, cap_for(args.data, args.systems, args.sample))
    run_baselines.main()


if __name__ == "__main__":
    main()
