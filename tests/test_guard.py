import json

import pytest

from gapdeleg.backends.base import UnguardedEgressError
from gapdeleg.pipeline.egress import GuardedCloud, load_outgoing
from gapdeleg.privacy.guard import EgressGuard, EgressTicket
from gapdeleg.privacy.types import Label, Span

from conftest import FakeCloud, FakeDetector

SPANS = [Span("Rachel Zheng", Label.LOCAL_ONLY, "PERSON"), Span("H&R Technology", Label.ABSTRACTABLE, "ORGANIZATION")]


def msg(text):
    return [{"role": "user", "content": text}]


def test_exact_and_fuzzy_local_only_denied(policy):
    g = EgressGuard(policy)
    assert not g.check(msg("Email for Rachel Zheng please"), SPANS).allow
    assert not g.check(msg("email for rachel  zheng"), SPANS).allow
    assert not g.check(msg("email for Rachel Zhengg"), SPANS).allow
    assert g.check(msg("email for [PERSON_1]"), SPANS).allow


def test_abstractable_span_is_not_a_guard_hit(policy):
    assert EgressGuard(policy).check(msg("works at H&R Technology"), SPANS).allow


def test_detector_pass_blocks_high_risk_types(policy):
    g = EgressGuard(policy, detector=FakeDetector({"rz@example.com": "EMAIL_ADDRESS", "Paris": "LOCATION"}))
    d = g.check(msg("write to rz@example.com"), [])
    assert not d.allow and "EMAIL_ADDRESS" in d.reasons[0]
    assert g.check(msg("a trip to Paris"), []).allow


def test_fail_closed_on_checker_error(policy):
    def boom(text, spans, request):
        raise RuntimeError("model crashed")
    d = EgressGuard(policy, llm_check=boom).check(msg("harmless"), [])
    assert not d.allow and "guard error" in d.reasons[0]


def test_llm_check_values_deny(policy):
    d = EgressGuard(policy, llm_check=lambda t, s, r: ["Erin"]).check(msg("a note for Erin"), [])
    assert not d.allow and d.reasons == ["local model: private value found"] and d.offending == ["Erin"]


def test_cloud_refuses_without_ticket():
    cloud = FakeCloud()
    with pytest.raises(UnguardedEgressError):
        cloud.generate(msg("hi"), ticket=None)
    forged = EgressTicket("0" * 64, "delegation")
    with pytest.raises(UnguardedEgressError):
        cloud.generate(msg("hi"), ticket=forged)
    assert cloud.calls == []


def test_ticket_bound_to_exact_messages(policy):
    g, cloud = EgressGuard(policy), FakeCloud()
    t = g.check(msg("ok text"), []).ticket
    with pytest.raises(UnguardedEgressError):
        cloud.generate(msg("different text"), ticket=t)
    assert cloud.generate(msg("ok text"), ticket=t).text == "cloud answer"


def test_guarded_cloud_logs_outgoing_and_decisions(tmp_path, policy):
    g = EgressGuard(policy, log_path=tmp_path / "guard.jsonl")
    gc = GuardedCloud(FakeCloud(), g, tmp_path / "outgoing.jsonl")
    assert gc.send(msg("about Rachel Zheng"), SPANS, request_id="r1") is None
    assert gc.send(msg("about [PERSON_1]"), SPANS, request_id="r1").text == "cloud answer"
    out = load_outgoing(tmp_path / "outgoing.jsonl")
    assert out == {"r1": ["about [PERSON_1]"]}
    decisions = [json.loads(x) for x in (tmp_path / "guard.jsonl").read_text().splitlines()]
    assert [d["allow"] for d in decisions] == [False, True]


def test_guard_disabled_allows_everything(policy):
    assert EgressGuard(policy, enabled=False).check(msg("Rachel Zheng"), SPANS).allow


def test_evaluation_ticket_works_and_is_logged(tmp_path, policy):
    g = EgressGuard(policy, log_path=tmp_path / "guard.jsonl")
    m = msg("judge this: Rachel Zheng")
    assert FakeCloud().generate(m, ticket=g.evaluation_ticket(m)).text == "cloud answer"
    assert json.loads((tmp_path / "guard.jsonl").read_text())["purpose"] == "evaluation"


def test_request_check_blocks_codes_the_labeler_missed(policy, tmp_path):
    g = EgressGuard(policy, request_check=True, log_path=tmp_path / "guard.jsonl")
    request = "My key is d2J3BgtYdB2 and my handle is @Xtension11. Fix my login script."
    d = g.check(msg("Fix the login script; the key is d2J3BgtYdB2"), [], request=request)
    assert not d.allow and d.offending == ["d2J3BgtYdB2"] and "ID_NUMBER" in d.reasons[0]
    assert g.check(msg("Fix the login script; the key is [ID_NUMBER_1]"), [], request=request).allow
    assert g.check(msg("the key is d2J3BgtYdB2"), []).allow
    assert EgressGuard(policy).check(msg("the key is d2J3BgtYdB2"), [], request=request).allow
    assert "d2J3BgtYdB2" not in (tmp_path / "guard.jsonl").read_text()


def test_offending_lists_values_from_every_check(policy):
    g = EgressGuard(policy, detector=FakeDetector({"rz@example.com": "EMAIL_ADDRESS"}), request_check=True,
                    llm_check=lambda t, s, r: [v for v in ["Erin"] if v in t])
    d = g.check(msg("Rachel Zheng, rz@example.com, ref 5551234567, for Erin"), SPANS,
                request="Rachel Zheng (ref 5551234567) writes to Erin")
    assert not d.allow and d.offending == ["Rachel Zheng", "rz@example.com", "5551234567", "Erin"]
    assert g.check(msg("harmless"), SPANS, request="Rachel Zheng (ref 5551234567)").offending == []


LEAK_REQUEST = 'Hi, I am Erin. Please write to koushik (user initi_user) about the roof. “Wow,” Andy said. 发邮件给Beckey'
LEAK_MESSAGE = "Write to [PERSON_1] about the roof. Please also write to koushik, initi_user and Beckey. Wow. Andy"


def test_leak_candidates_are_names_and_rare_words_the_message_still_has():
    from gapdeleg.privacy.guard import leak_candidates
    assert leak_candidates(LEAK_MESSAGE, LEAK_REQUEST) == ["initi_user", "Beckey", "koushik", "Andy"]
    assert leak_candidates(LEAK_MESSAGE, LEAK_REQUEST, limit=2) == ["initi_user", "Beckey"]
    assert leak_candidates("Write to [PERSON_1] about the roof.", LEAK_REQUEST) == []
    assert leak_candidates("Ask Andy about koushik") == ["koushik", "Andy"]


def test_edge_leak_check_judges_the_candidates():
    from gapdeleg.privacy.guard import EdgeLeakCheck
    from conftest import FakeEdge
    edge = FakeEdge(["<think></think>1: no\n2. YES\n3: no"])
    check = EdgeLeakCheck(edge)
    found = check(LEAK_MESSAGE, [], request=LEAK_REQUEST)
    assert found == ["Beckey", "Andy"]
    assert check.last == {"candidates": ["initi_user", "Beckey", "koushik", "Andy"], "private": ["Beckey", "Andy"],
                          "unanswered": 1, "answer": "<think></think>1: no\n2. YES\n3: no"}
    prompt = edge.calls[0][0]["content"]
    assert LEAK_REQUEST in prompt and "1. initi_user\n2. Beckey\n3. koushik\n4. Andy" in prompt
    assert check("Write to [PERSON_1] about the roof.", [], request=LEAK_REQUEST) == []
    assert len(edge.calls) == 1
    assert len(check.drain_generations()) == 1 and check.drain_generations() == []


def test_edge_leak_check_in_chunks():
    from gapdeleg.privacy.guard import EdgeLeakCheck
    from conftest import FakeEdge
    edge = FakeEdge(["1: no\n2: yes", "1: yes\n2: no"])
    check = EdgeLeakCheck(edge, chunk=2)
    assert check(LEAK_MESSAGE, [], request=LEAK_REQUEST) == ["Beckey", "koushik"]
    assert len(edge.calls) == 2 and "1. koushik\n2. Andy" in edge.calls[1][0]["content"]
    assert check.last["unanswered"] == 0 and check.last["answer"] == "1: no\n2: yes\n---\n1: yes\n2: no"
    assert len(check.drain_generations()) == 2


def test_leak_check_wordings_and_answer_formats():
    from gapdeleg.privacy.guard import EdgeLeakCheck, _verdicts
    from conftest import FakeEdge
    assert _verdicts("1: no\n2. Beckey: YES\n3) **koushik** - no\n2: no\nnote 4", {"yes", "no"}) == \
        {1: "no", 2: "yes", 3: "no"}
    assert _verdicts("1: CARBS / personal\n2: click /no\n3: arm64-v8a /no", {"personal", "no"}) == \
        {1: "personal", 2: "no", 3: "no"}
    labels = EdgeLeakCheck(FakeEdge(["1: no\n2: name\n3: personal\n4: yes"]), wording="labels")
    assert labels(LEAK_MESSAGE, [], request=LEAK_REQUEST) == ["Beckey", "koushik", "Andy"]
    safe = EdgeLeakCheck(FakeEdge(["1: yes\n2: no\n3: yes\n4: yes"]), wording="safe")
    assert safe(LEAK_MESSAGE, [], request=LEAK_REQUEST) == ["Beckey"]
    with pytest.raises(ValueError):
        EdgeLeakCheck(FakeEdge([]), wording="nope")
