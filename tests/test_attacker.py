import json

from gapdeleg.metrics.attacker import Attacker, aggregate, attack_targets, matches
from gapdeleg.privacy.guard import EgressGuard
from gapdeleg.privacy.types import Label, Span

from conftest import FakeCloud

SPANS = [Span("Maya Okafor", Label.LOCAL_ONLY, "PERSON"), Span("Maya", Label.LOCAL_ONLY, "PERSON"),
         Span("Brightline Logistics", Label.LOCAL_ONLY, "ORGANIZATION"),
         Span("Leeds", Label.ABSTRACTABLE, "LOCATION")]


def test_targets_are_distinct_local_only_values():
    t = attack_targets(SPANS)
    assert [s.text for s in t] == ["Brightline Logistics", "Maya Okafor"]


def test_matches():
    assert matches("Maya Okafor", "maya okafor")
    assert matches("Brightline Logistics", "Brightline Logistics Ltd")
    assert matches("4417-2290-8812", "account 4417-2290-8812")
    assert not matches("Maya Okafor", "Maya")
    assert not matches("Maya Okafor", "")
    assert not matches("Brightline Logistics", "a logistics company")


def fake_attacker(guess_fn):
    def reply(messages):
        return json.dumps({"guesses": guess_fn(messages[0]["content"])})
    return Attacker(FakeCloud(reply=reply), EgressGuard())


def test_attack_scores_against_truth_and_prior():
    def guesses(prompt):
        if "(no messages were sent)" in prompt:
            return {"0": "Acme Corp", "1": "John Smith"}
        if "Brightline" in prompt:
            return {"0": "Brightline Logistics", "1": "John Smith"}
        return {"0": "Acme Corp", "1": "Jane Doe"}
    att = fake_attacker(guesses)
    r = att.attack("r1", SPANS, ["I work at Brightline Logistics as [PERSON_1]"])
    assert [(a.text, a.reconstructed, a.prior_reconstructed) for a in r.attributes] == [
        ("Brightline Logistics", True, False), ("Maya Okafor", False, False)]
    assert r.rate == 0.5 and r.prior_rate == 0.0
    clean = att.attack("r2", SPANS, ["I work at [ORGANIZATION_1] as [PERSON_1]"])
    assert clean.rate == 0.0
    agg = aggregate([r, clean])
    assert agg["attributes"] == 4 and agg["reconstruction_rate"] == 0.25
    assert agg["requests_with_any_reconstruction"] == 1
    assert agg["by_category"]["ORGANIZATION"] == {"n": 2, "reconstructed": 1, "prior_reconstructed": 0}


def test_same_category_guesses_match_any_value_of_that_category():
    from gapdeleg.metrics.attacker import match_within_category, rescore
    names = [("Laila", "PERSON"), ("Omar", "PERSON"), ("Rahimi Sahib", "PERSON")]
    assert match_within_category(names, ["Rahimi Sahib", "Laila Khan", "Omar"]) == [True, True, True]
    assert match_within_category(names, ["Laila", "Laila", "x"]) == [True, False, False]
    assert match_within_category([("Laila", "PERSON"), ("Acme", "ORGANIZATION")], ["Acme", "Laila"]) == [False, False]
    stored = [{"text": t, "category": c, "guess": g, "prior_guess": "", "reconstructed": False,
               "prior_reconstructed": False} for (t, c), g in zip(names, ["Omar", "Laila", "y"])]
    assert [a["reconstructed"] for a in rescore(stored)] == [True, True, False]


def test_no_outgoing_still_attacked_from_prior_only():
    att = fake_attacker(lambda p: {"0": "Brightline Logistics", "1": "x"})
    r = att.attack("r", SPANS, [])
    assert r.attributes[0].reconstructed and r.attributes[0].prior_reconstructed


def test_unparseable_attacker_output_counts_as_no_reconstruction():
    att = Attacker(FakeCloud(reply="I refuse"), EgressGuard())
    r = att.attack("r", SPANS, ["Brightline Logistics"])
    assert r.rate == 0.0 and all(a.guess == "" for a in r.attributes)


def test_no_local_only_spans_no_calls():
    cloud = FakeCloud()
    r = Attacker(cloud, EgressGuard()).attack("r", [Span("Leeds", Label.ABSTRACTABLE, "LOCATION")], ["Leeds"])
    assert r.attributes == [] and cloud.calls == []
