import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import famb_baselines
import famb_compare
import famb_method
import famb_run
from famb_common import EDGES

from gapdeleg import experiment
from gapdeleg.privacy.types import Label, Span

ENV = {"OLLAMA_HOST_URL": "http://127.0.0.1:12345"}


def build(*argv):
    return famb_method.build(famb_method.parser().parse_args(list(argv)), environ=ENV)


def test_the_systems_are_the_chosen_ones():
    assert build("--data", "privfin_test", "--edge", EDGES[0])[0]["system"] == "gapdeleg.formula.formula:make"
    cfg = build("--data", "privmed_test", "--edge", EDGES[0])[0]
    assert cfg["system"] == "method_request" and cfg["system_args"] == {"confidence_gate": False}


def test_the_check_on_dev_is_small_and_goes_to_its_own_folder():
    cfg, folder = build("--data", "privfin_dev", "--edge", EDGES[0], "--limit", "5")
    assert folder.name == "quick" and len(cfg["data"]["ids"]) == 5 and cfg["cloud_budget_usd"] == 0.02


@pytest.mark.parametrize("argv", [
    ["--data", "privfin_test", "--edge", "qwen3:1.7b", "--limit", "5"],
    ["--data", "privfin_dev", "--edge", "qwen3:1.7b"],
    ["--data", "privfin_dev", "--edge", "qwen3:1.7b", "--limit", "6"],
    ["--data", "privmed_test", "--edge", "qwen3:1.7b", "--cap", "0.31"],
    ["--data", "privmed_dev", "--edge", "qwen3:1.7b", "--limit", "5", "--cap", "0.1"],
])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        build(*argv)


def span(query, text):
    start = query.find(text)
    return Span(text=text, label=Label.LOCAL_ONLY, category="MEDICAL_CONDITION", start=start, end=start + len(text))


QUERY = "I feel tired and cold and I have a fever. Which of these: common cold, malaria?"
SPANS = [span(QUERY, "cold"), span(QUERY, "cold"), span(QUERY, "fever")]
EXAMPLES = [SimpleNamespace(id=i, query=QUERY, meta={"options": ["common cold", "malaria"]}, private_spans=SPANS)
            for i in ("list-only", "symptom", "not-run")]


def write_run(path, stored):
    path.mkdir()
    (path / "results.jsonl").write_text("\n".join(json.dumps({"id": i, "exact_match": i == "symptom"})
                                                  for i in ("list-only", "symptom")))
    (path / "leakage.json").write_text(json.dumps([{"request_id": i, "hard_violation": v} for i, v in stored.items()]))
    (path / "outgoing.jsonl").write_text("\n".join(json.dumps(
        {"request_id": i, "step": "cloud", "model": "m", "messages": [{"role": "user", "content": text}]})
        for i, text in (("list-only", "Tell apart: common cold, malaria."), ("symptom", "I feel cold and I have a fever."))))


def test_outcomes_count_a_loss_and_both_counts(tmp_path):
    write_run(tmp_path / "run", {"list-only": True, "symptom": True})
    out, mismatch = famb_compare.outcomes(tmp_path / "run", EXAMPLES)
    assert out["not-run"] == (0, 0, 0)
    assert out["symptom"] == (1, 1, 1)
    assert out["list-only"] == (0, 1, 0)
    assert mismatch == 0


def test_a_stored_result_that_differs_from_the_rerun_is_flagged(tmp_path):
    write_run(tmp_path / "run", {"list-only": True, "symptom": False})
    assert famb_compare.outcomes(tmp_path / "run", EXAMPLES)[1] == 1


def cell(edge="e", counted=True, hard=0, diff=0.1, p=0.01, stage_2=True):
    test = {"baseline": "edge_only", "diff": diff, "p": p}
    return {"data": "d", "edge": edge, "counted": counted, "hard_measured": hard, "hard_rule": hard,
            "vs_best": {"hard_measured": test, "hard_rule": test}, "stage_2_done": stage_2}


@pytest.mark.parametrize("c, ok", [(cell(), True), (cell(hard=1), False), (cell(p=0.06), False), (cell(diff=-0.1), False)])
def test_a_pair_needs_zero_violations_and_a_significant_win(c, ok):
    assert famb_compare.meets(c, "hard_rule") is ok


def test_the_rule_needs_4_of_8_and_a_pair_outside_the_count_does_not_help():
    assert sum(len(v) for v in famb_compare.COUNTED.values()) == 8
    three = [cell() for _ in range(3)] + [cell(p=0.5) for _ in range(5)]
    v = famb_compare.verdict(three + [cell(counted=False)], "hard_rule")
    assert (v["counted"], v["need"], v["passed"], v["state"]) == (8, 4, 3, "no_go")
    four = [cell() for _ in range(4)] + [cell(p=0.5) for _ in range(4)]
    assert famb_compare.verdict(four, "hard_rule")["state"] == "go"
    assert famb_compare.verdict(four[:-1] + [{**cell(), "missing": True}], "hard_rule")["state"] == "incomplete"


def test_without_stage_2_four_pairs_are_not_a_go_but_three_are_still_a_no_go():
    four = [cell(stage_2=False) for _ in range(4)] + [cell(p=0.5, stage_2=False) for _ in range(4)]
    assert famb_compare.verdict(four, "hard_rule")["state"] == "needs_stage_2"
    three = [cell(stage_2=False) for _ in range(3)] + [cell(p=0.5, stage_2=False) for _ in range(5)]
    assert famb_compare.verdict(three, "hard_rule")["state"] == "no_go"


def test_a_masked_option_can_only_take_a_pair_out_of_the_extra_count():
    five = [cell() for _ in range(5)] + [cell(p=0.5) for _ in range(3)]
    assert famb_compare.verdict_if_masked_counted(five)["passed"] == 5
    masked = [{**five[0], "masked_options": 5}] + five[1:]
    v = famb_compare.verdict_if_masked_counted(masked)
    assert (v["passed"], v["state"]) == (4, "go")
    assert famb_compare.verdict(masked, "hard_rule")["passed"] == 5


def test_strict_counts_a_symptom_alone_and_the_list_only_as_measured(tmp_path):
    write_run(tmp_path / "run", {"list-only": True, "symptom": True})
    assert famb_compare.strict_counts(tmp_path / "run", EXAMPLES) == {"strict_measured": 2, "strict_rule": 1}


def test_a_strict_match_takes_a_pair_out_of_the_strict_count_only():
    five = [cell() for _ in range(5)] + [cell(p=0.5) for _ in range(3)]
    some = [{**c, "strict_rule": 1} for c in five[:3]] + five[3:]
    v = famb_compare.verdict_if_strict(some, "strict_rule")
    assert (v["passed"], v["state"]) == (2, "no_go")
    assert famb_compare.verdict(some, "hard_rule")["passed"] == 5


def test_the_bound_takes_redact_and_send_only_where_it_does_not_leak():
    edge = {"wrong, leaks": (0, 0, 0), "wrong, clean": (0, 0, 0), "right": (1, 0, 0)}
    redact = {"wrong, leaks": (1, 1, 1), "wrong, clean": (1, 1, 0), "right": (0, 0, 0)}
    assert famb_compare.zero_violation_bound(edge, redact) == {"wrong, leaks": (0,), "wrong, clean": (1,), "right": (1,)}


def test_the_bar_is_the_best_baseline_with_zero_violations_under_that_count():
    base = {"edge_only": {"exact": 0.5, "hard_measured": 0, "hard_rule": 0},
            "redact_send": {"exact": 0.7, "hard_measured": 1, "hard_rule": 0},
            "cloud_full": {"exact": 0.9, "hard_measured": 5, "hard_rule": 5}}
    assert famb_compare.best_baseline(base, "hard_measured") == "edge_only"
    assert famb_compare.best_baseline(base, "hard_rule") == "redact_send"
    base["redact_send"]["exact"] = 0.5
    assert famb_compare.best_baseline(base, "hard_rule") == "edge_only"


@pytest.mark.parametrize("data, system, sample", [
    ("privfin_test", "redact_send", 5),
    ("privfin_test", "redact_send", None),
    ("privfin_test", "edge_only,redact_send", 0),
    ("privfin_test", "papillon", 0),
    ("privfin_dev", "redact_send", 0),
    ("privfin_dev", "redact_send", 6),
    ("pupa_tnb", "redact_send", 0),
])
def test_the_capped_baseline_runner_refuses_bad_arguments(data, system, sample):
    with pytest.raises(SystemExit):
        famb_baselines.cap_for(data, system, sample)


def test_the_capped_runner_gives_run_baselines_config_only_a_cap(tmp_path, monkeypatch):
    seen = []

    def fake_run(cfg, *args, **kw):
        seen.append(cfg)
        raise RuntimeError("stop here")

    def config(script_main, cap_wrapped):
        monkeypatch.setattr(experiment, "run", fake_run)
        monkeypatch.setattr(sys, "argv", [script_main, "--data", "privfin_dev", "--sample", "5", "--systems", "redact_send",
                                          "--edge", "qwen3:1.7b", "--no-attacker", "--no-judge", "--tag", "check",
                                          "--out", str(tmp_path / script_main)])
        cap_wrapped()
        return seen[-1]

    plain = config("plain", famb_baselines.run_baselines.main)
    capped = config("capped", famb_baselines.main)
    assert capped["cloud_budget_usd"] == famb_baselines.CHECK_CAP and "cloud_budget_usd" not in plain
    assert {k: v for k, v in capped.items() if k != "cloud_budget_usd"} == plain
    assert plain["system"] == "redact_send" and plain["labeler"] == ["detector", "edge"]
    assert plain["judge"] is None and plain["attacker"] is None and plain["cloud_cache"] == "runs/cache"
