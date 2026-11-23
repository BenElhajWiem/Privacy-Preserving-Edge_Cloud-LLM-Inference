import json
from pathlib import Path

import numpy as np

from gapdeleg.metrics.quality import is_empty_loss

METRICS = {
    "at_least_as_good": ("good", "judged"),
    "hard_violation": ("hard", None),
    "hard_violation_strict": ("hard_strict", None),
    "span_leak_local_only": ("local_leaked", "local_n"),
    "abstraction_compliance": ("abs_kept", "abs_n"),
    "attacker_reconstruction": ("reconstructed", "attacked"),
    "cloud_calls": ("cloud_calls", None),
    "cloud_in_tokens": ("cloud_in", None),
    "cloud_out_tokens": ("cloud_out", None),
}
FIELDS = sorted({k for pair in METRICS.values() for k in pair if k})


def request_outcomes(run: Path) -> dict[str, dict]:
    def by_request(name):
        p = run / name
        return {x["request_id"]: x for x in json.loads(p.read_text())} if p.exists() else {}
    leak, att = by_request("leakage.json"), by_request("attacker.json")
    rows = {}
    recs = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]
    judged_run = any(r.get("judge") for r in recs)
    for r in recs:
        spans = leak.get(r["id"], {}).get("spans", [])
        local = [s for s in spans if s["label"] == "local-only"]
        abstr = [s for s in spans if s["label"] == "abstractable"]
        attrs = att.get(r["id"], {}).get("attributes", [])
        verdict = (r.get("judge") or {}).get("verdict")
        rows[r["id"]] = {
            "judged": int(verdict is not None or (judged_run and is_empty_loss(r))), "good": int(verdict in ("win", "tie")),
            "hard": int(any(s["leaked"] for s in local)), "hard_strict": int(any(s["leaked_strict"] for s in local)),
            "local_n": len(local), "local_leaked": sum(s["leaked"] for s in local),
            "abs_n": len(abstr), "abs_kept": sum(not s["leaked"] for s in abstr),
            "attacked": len(attrs), "reconstructed": sum(a["reconstructed"] for a in attrs),
            "cloud_calls": r["cost"]["cloud_calls"], "cloud_in": r["cost"]["cloud_in_tokens"],
            "cloud_out": r["cost"]["cloud_out_tokens"]}
    return rows


def point_values(rows: list[dict]) -> dict[str, float | None]:
    out = {}
    for m, (num, den) in METRICS.items():
        d = sum(r[den] for r in rows) if den else len(rows)
        out[m] = sum(r[num] for r in rows) / d if d else None
    return out


def _arrays(runs: list[dict[str, dict]], ids: list[str]) -> list[dict[str, np.ndarray]]:
    return [{f: np.array([run[i][f] for i in ids], dtype=float) for f in FIELDS} for run in runs]


def _values(arrays: list[dict[str, np.ndarray]], idx: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for m, (num, den) in METRICS.items():
            per_run = [a[num][idx].sum(axis=-1) / (a[den][idx].sum(axis=-1) if den else idx.shape[-1]) for a in arrays]
            out[m] = np.mean(per_run, axis=0)
    return out


def resamples(n: int, b: int = 10_000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def confidence_intervals(runs: list[dict[str, dict]], ids: list[str], idx: np.ndarray) -> dict[str, dict]:
    arrays = _arrays(runs, ids)
    point = _values(arrays, np.arange(len(ids)))
    boot = _values(arrays, idx)
    return {m: {"value": float(point[m]), "low": float(np.nanpercentile(boot[m], 2.5)),
                "high": float(np.nanpercentile(boot[m], 97.5))} for m in METRICS}


def paired_test(runs_a: list[dict[str, dict]], runs_b: list[dict[str, dict]], ids: list[str],
                idx: np.ndarray) -> dict[str, dict]:
    a, b = _arrays(runs_a, ids), _arrays(runs_b, ids)
    full = np.arange(len(ids))
    pa, pb, ba, bb = _values(a, full), _values(b, full), _values(a, idx), _values(b, idx)
    out = {}
    for m in METRICS:
        d = ba[m] - bb[m]
        d = d[~np.isnan(d)]
        p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))) if len(d) else float("nan")
        out[m] = {"diff": float(pa[m] - pb[m]), "low": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
                  "high": float(np.percentile(d, 97.5)) if len(d) else float("nan"), "p": p}
    return out
