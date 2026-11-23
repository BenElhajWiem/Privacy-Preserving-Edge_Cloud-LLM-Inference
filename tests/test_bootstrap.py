import json

from gapdeleg.metrics.bootstrap import confidence_intervals, paired_test, request_outcomes, resamples


def write_run(path, requests):
    path.mkdir()
    cost = {"cloud_calls": 1, "cloud_in_tokens": 100, "cloud_out_tokens": 50}

    def rec(i, v):
        r = {"id": i, "answer": "" if v == "empty" else "an answer", "cost": cost}
        if v == "refused":
            r["judge_error"] = "BadRequestError: flagged"
        elif v and v != "empty":
            r["judge"] = {"verdict": v}
        return r
    (path / "results.jsonl").write_text("\n".join(json.dumps(rec(i, v)) for i, v, _, _ in requests))
    (path / "leakage.json").write_text(json.dumps([
        {"request_id": i, "spans": [{"label": "local-only", "leaked": x, "leaked_strict": x} for x in leaks] +
                                   [{"label": "abstractable", "leaked": False, "leaked_strict": False}]}
        for i, _, leaks, _ in requests]))
    (path / "attacker.json").write_text(json.dumps([
        {"request_id": i, "attributes": [{"reconstructed": x} for x in rec]} for i, _, _, rec in requests]))
    return request_outcomes(path)


def test_point_estimates_pool_spans_and_skip_judge_refusals(tmp_path):
    run = write_run(tmp_path / "a", [("r-0", "win", [True, True, True], [True]),
                                     ("r-1", "loss", [False], [False, False, False]),
                                     ("r-2", "refused", [False], [])])
    ids = ["r-0", "r-1", "r-2"]
    ci = confidence_intervals([run], ids, resamples(len(ids), 500))
    assert ci["at_least_as_good"]["value"] == 0.5
    assert abs(ci["hard_violation"]["value"] - 1 / 3) < 1e-12
    assert ci["span_leak_local_only"]["value"] == 3 / 5
    assert ci["attacker_reconstruction"]["value"] == 1 / 4
    assert ci["abstraction_compliance"]["value"] == 1.0
    for m in ci.values():
        assert m["low"] <= m["value"] <= m["high"]


def test_seeds_are_averaged_on_the_same_resample(tmp_path):
    a = write_run(tmp_path / "a", [("r-0", "win", [], []), ("r-1", "loss", [], [])])
    b = write_run(tmp_path / "b", [("r-0", "loss", [], []), ("r-1", "loss", [], [])])
    ci = confidence_intervals([a, b], ["r-0", "r-1"], resamples(2, 200))
    assert ci["at_least_as_good"]["value"] == 0.25


def test_paired_test(tmp_path):
    reqs = [(f"r-{k}", "win" if k < 10 else "loss", [k < 5], []) for k in range(40)]
    a = write_run(tmp_path / "a", reqs)
    same = paired_test([a], [a], [r[0] for r in reqs], resamples(40, 1000))
    assert same["at_least_as_good"]["diff"] == 0 and same["at_least_as_good"]["p"] == 1.0
    worse = write_run(tmp_path / "b", [(i, "loss", [True], []) for i, *_ in reqs])
    t = paired_test([a], [worse], [r[0] for r in reqs], resamples(40, 1000))
    assert t["at_least_as_good"]["diff"] == 0.25 and t["at_least_as_good"]["p"] < 0.01
    assert t["hard_violation"]["diff"] < 0 and t["hard_violation"]["high"] < 0


def test_empty_answers_count_as_losses(tmp_path):
    from gapdeleg.metrics.quality import quality_summary
    reqs = [("r-0", "win", [], []), ("r-1", "tie", [], []), ("r-2", "loss", [], []), ("r-3", "empty", [], []),
            ("r-4", "refused", [], [])]
    run = write_run(tmp_path / "a", reqs)
    ids = [r[0] for r in reqs]
    assert confidence_intervals([run], ids, resamples(len(ids), 200))["at_least_as_good"]["value"] == 2 / 4
    recs = [json.loads(x) for x in (tmp_path / "a" / "results.jsonl").read_text().splitlines()]
    q = quality_summary(recs)
    assert q["at_least_as_good"] == 2 / 4 and q["empty_answers_as_loss"] == 1 and q["judge_errors"] == 1
    assert quality_summary([{"id": "x", "answer": ""}]) is None
