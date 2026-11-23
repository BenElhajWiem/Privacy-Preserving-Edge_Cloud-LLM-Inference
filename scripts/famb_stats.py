import numpy as np

B, SEED = 10_000, 0


def paired(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    x = np.array([a[i][0] for i in ids], float)
    y = np.array([b[i][0] for i in ids], float)
    idx = np.random.default_rng(SEED).integers(0, len(ids), size=(B, len(ids)))
    d = x[idx].mean(1) - y[idx].mean(1)
    p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
    return {"n": len(ids), "a": float(x.mean()), "b": float(y.mean()), "diff": float(x.mean() - y.mean()),
            "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)), "p": p}
