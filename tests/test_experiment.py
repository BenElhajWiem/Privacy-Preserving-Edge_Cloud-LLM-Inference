import json

import gapdeleg.experiment as experiment
from gapdeleg.build.io import read_examples, request_ids

from conftest import FakeCloud, FakeEdge


def test_judge_refusal_leaves_the_request_unjudged_and_the_run_continues(tmp_path, monkeypatch):
    ids = request_ids("pupa_tnb")[:2]
    refused = {e.id: e for e in read_examples("pupa_tnb")}[ids[0]].query

    def judge(messages):
        if refused in messages[0]["content"]:
            raise RuntimeError("This content was flagged")
        return "[[A]]"
    monkeypatch.setattr(experiment, "make_edge", lambda cfg: FakeEdge())
    monkeypatch.setattr(experiment, "make_cloud", lambda cfg, cache_dir=None: FakeCloud(reply=judge))
    rd = experiment.run({"name": "t", "system": "edge_only", "edge": {}, "judge": {"backend": "fake"},
                         "data": {"dataset": "processed", "name": "pupa_tnb", "ids": ids}},
                        run_root=tmp_path, log=lambda *a: None)
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
    assert [r["id"] for r in recs] == ids
    assert "flagged" in recs[0]["judge_error"] and "judge" not in recs[0]
    assert recs[1]["judge"]["verdict"] in ("win", "tie", "loss")
    quality = json.loads((rd.path / "summary.json").read_text())["quality"]
    assert quality["judge_errors"] == 1 and sum(quality["verdicts"].values()) == 1


def test_cloud_budget_stops_the_run_and_spend_is_reported(tmp_path, monkeypatch):
    from gapdeleg.backends import Generation, Usage

    class PricedCloud(FakeCloud):
        name = "claude-sonnet-5-5"

        def _generate(self, messages, **params):
            return Generation("cloud answer", usage=Usage(0, 200_000), model=self.name)
    ids = request_ids("pupa_tnb")[:3]
    monkeypatch.setattr(experiment, "make_edge", lambda cfg: FakeEdge())
    monkeypatch.setattr(experiment, "make_cloud", lambda cfg, cache_dir=None: PricedCloud())
    rd = experiment.run({"name": "t", "system": "cloud_full", "edge": {}, "cloud": {"backend": "fake"},
                         "cloud_budget_usd": 1.0, "data": {"dataset": "processed", "name": "pupa_tnb", "ids": ids}},
                        run_root=tmp_path, log=lambda *a: None)
    s = json.loads((rd.path / "summary.json").read_text())
    assert s["n"] == 1 and s["stopped_at_budget"] == ids[1] and s["cloud_spend_usd"] == 2.0
    assert len(rd.results.read_text().splitlines()) == 1


def test_method_curves_route_by_the_open_runs_confidence():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from method_curves import point, select

    def outcome(good, hard):
        return {"judged": 1, "good": good, "hard": hard, "hard_strict": hard, "local_n": 1, "local_leaked": hard,
                "abs_n": 0, "abs_kept": 0, "attacked": 0, "reconstructed": 0, "cloud_calls": 1, "cloud_in": 1,
                "cloud_out": 1}
    ids = ["a", "b", "c"]
    conf = {"a": 0.2, "b": 0.5, "c": 0.9}
    open_ = {"outcomes": {"a": outcome(1, 0), "b": outcome(1, 1), "c": outcome(1, 0)}}
    closed = {"outcomes": {"a": outcome(0, 0), "b": outcome(0, 0), "c": outcome(0, 0)}}
    assert point(ids, conf, 0.3, open_, closed)["escalated"] == 1 / 3
    assert point(ids, conf, 1.01, open_, closed)["at_least_as_good"] == 1.0
    sel = select(ids, conf, open_, closed)
    assert sel["best_quality"]["at_least_as_good"] == 1.0 and sel["best_quality"]["hard_violation"] == 1 / 3
    assert sel["zero_violation"]["threshold"] == 0.35 and sel["zero_violation"]["at_least_as_good"] == 1 / 3
