from gapdeleg.metrics.leakage import aggregate, request_leakage
from gapdeleg.privacy.matching import exact_match, span_found
from gapdeleg.privacy.types import Label, Span

LO, AB = Label.LOCAL_ONLY, Label.ABSTRACTABLE


def test_matching():
    assert exact_match("Rachel", "dear rachel,")
    assert not exact_match("al", "also")
    assert span_found("univeristy of westminster", "the University of Westminster hotel") == "fuzzy"
    assert span_found("Bob", "Bobby") is None


def test_union_of_messages_counts():
    spans = [Span("Rachel Zheng", LO), Span("H&R Technology", AB)]
    r = request_leakage("r", spans, ["no names here", "but here: rachel zheng"])
    assert r.hard_violation
    assert [s.leaked for s in r.spans] == [True, False]


def test_no_outgoing_no_leak():
    r = request_leakage("r", [Span("Rachel", LO)], [])
    assert not r.hard_violation and r.n_outgoing == 0


def test_abstractable_leak_is_not_hard_violation():
    r = request_leakage("r", [Span("Paris", AB)], ["trip to Paris"])
    assert not r.hard_violation and r.spans[0].leaked


def test_aggregate():
    reqs = [request_leakage("a", [Span("Rachel", LO), Span("Paris", AB)], ["Rachel in Paris"]),
            request_leakage("b", [Span("Johnny", LO)], ["[PERSON_1]"]),
            request_leakage("c", [Span("Ann", LO)], [])]
    a = aggregate(reqs)
    assert a["n_requests"] == 3 and a["requests_with_outgoing"] == 2
    assert abs(a["hard_violation_rate"] - 1 / 3) < 1e-9
    assert a["n_spans"] == 4 and a["span_leakage_rate"] == 0.5
    assert a["per_label"]["local-only"] == {"spans": 3, "leaked": 1, "rate": 1 / 3, "rate_strict": 1 / 3,
                                            "exact_only_rate": 1 / 3}
    assert a["per_label"]["abstractable"]["rate"] == 1.0


def test_weak_quantities_need_context():
    q = "I paid a $60 fee from my account and bought 3 tickets."
    s60 = Span("$60", LO, "MONEY", start=9, end=12)
    s3 = Span("3", LO, "AMOUNT", start=44, end=45)
    r = request_leakage("r", [s60, s3], ["Multiply the price by 3, then subtract $60 is not needed"], query=q)
    assert [x.leaked_strict for x in r.spans] == [True, True]
    assert [x.leaked for x in r.spans] == [False, False]
    assert r.hard_violation_strict and not r.hard_violation
    r = request_leakage("r", [s60, s3], ["I paid a $60 fee"], query=q)
    assert r.spans[0].leaked and r.spans[0].how == "context"
    assert request_leakage("r", [s3], ["3"]).spans[0].leaked


def test_weak_symptoms_need_another_symptom():
    q = "I have chills and nausea and a rash."
    sp = [Span("chills", LO, "MEDICAL_CONDITION", start=7, end=13),
          Span("nausea", LO, "MEDICAL_CONDITION", start=18, end=24)]
    assert not request_leakage("r", sp, ["What usually causes nausea?"], query=q).hard_violation
    assert request_leakage("r", sp, ["Patient has chills and nausea"], query=q).hard_violation


def test_strong_spans_unchanged_by_context_rule():
    q = "Rachel paid $60"
    r = request_leakage("r", [Span("Rachel", LO, "PERSON", start=0, end=6)], ["Rachel"], query=q)
    assert r.hard_violation and r.hard_violation_strict
