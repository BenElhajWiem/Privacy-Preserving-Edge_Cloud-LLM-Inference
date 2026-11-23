import math

import pytest

from gapdeleg.backends.base import Generation, Usage
from gapdeleg.baselines import CloudFull, EdgeOnly, RedactSend
from gapdeleg.confidence import MeanLogprob, mean_logprob, min_logprob
from gapdeleg.data.schema import Example
from gapdeleg.metrics.cost import aggregate, request_cost
from gapdeleg.metrics.quality import (PairwiseJudge, answer_line_number, at_least_as_good_rate, chosen_option,
                                      extract_number, numeric_exact_match, option_exact_match)
from gapdeleg.pipeline.context import Context, Trace, strip_think
from gapdeleg.pipeline.egress import GuardedCloud
from gapdeleg.pipeline.run import Flags, GapAwareDelegation
from gapdeleg.pipeline.gap_report import GapReport
from gapdeleg.privacy.guard import EgressGuard
from gapdeleg.privacy.labeler import parse_json_list
from gapdeleg.privacy.types import Label, Span

from conftest import FakeCloud, FakeEdge


class StaticLabeler:
    def __init__(self, spans):
        self.spans = spans

    def label(self, text):
        return self.spans


def test_cache_hit(tmp_path, policy):
    cloud, g = FakeCloud(cache_dir=tmp_path), EgressGuard(policy)
    m = [{"role": "user", "content": "q"}]
    a = cloud.generate(m, ticket=g.check(m, []).ticket)
    b = cloud.generate(m, ticket=g.check(m, []).ticket)
    assert len(cloud.calls) == 1 and not a.cached and b.cached and b.text == a.text
    cloud.generate(m, ticket=g.check(m, []).ticket, max_tokens=7)
    assert len(cloud.calls) == 2


def test_confidence_logprob_ignores_think_block():
    g = Generation("<think>x</think>42", token_logprobs=[-5.0, -5.0, -5.0, -0.1],
                   tokens=["<think>", "x", "</think>", "42"])
    assert mean_logprob(g) == pytest.approx(-0.1)
    assert min_logprob(Generation("ab", [-1.0, -3.0])) == -3.0
    assert MeanLogprob().score("q", Generation("ab", [-1.0, -1.0])) == pytest.approx(math.exp(-1))
    with pytest.raises(ValueError):
        mean_logprob(Generation("x"))


def test_gap_report_parse():
    s = GapReport.parse('<think>hm</think>```json\n{"established": ["tone"], "uncertain": '
                             '[{"text": "hotel prices", "may_leave": true}], "missing": '
                             '[{"text": "her manager\'s name", "may_leave": false}]}\n```')
    assert [i.text for i in s.established] == ["tone"]
    assert s.gap_requests() == ["hotel prices"]
    assert GapReport.parse("not json").gap_requests() == []


def test_quality_helpers(policy):
    assert numeric_exact_match("so the answer is 1,250.", "1250")
    assert not numeric_exact_match("no number", "3")
    assert answer_line_number("Total $588.\n\n**Answer: $588**\n\nAccount 2722-1329-3015.") == "588"
    assert answer_line_number("### Final Answer: 1,320") == "1320"
    assert answer_line_number("The answer: Emma is 47.") is None
    assert numeric_exact_match("Answer: 64\n\nTip: keep account 5621-3068-6403 private.", "64")
    assert extract_number(r"so $$\boxed{17}$$ cans, from 3 pumpkins") == "17"
    assert extract_number("**Total: $135.00**\n\nNote: you don't need to share account 12.") == "135.00"
    assert extract_number("Step 2: 30 x 12 = 360\n### Final Answer:\nYou save **$120 per year**.") == "120"
    judge = PairwiseJudge(FakeCloud(reply=lambda m: "A"), EgressGuard(policy))
    r = judge.compare("q", "cand", "ref")
    assert (r.order_ab, r.order_ba, r.verdict) == ("A", "A", "tie")
    assert at_least_as_good_rate([r]) == 1.0


def test_option_exact_match_picks_the_named_diagnosis():
    opts = ["allergy", "chicken pox", "fungal infection", "gastroesophageal reflux disease", "impetigo",
            "psoriasis"]
    assert chosen_option("<think>maybe psoriasis</think>The most likely diagnosis is **fungal infection**.",
                         opts) == "fungal infection"
    assert chosen_option("It is not allergy. Fungal infection is the most likely cause. Psoriasis is less "
                         "likely.", opts) == "fungal infection"
    assert chosen_option("Allergy fits less well than psoriasis.\n### Final Answer: **Impetigo**", opts) == "impetigo"
    assert chosen_option("Psoriasis.", opts) == "psoriasis"
    assert chosen_option("Sounds like chickenpox.", opts) == "chicken pox"
    assert chosen_option("This looks like GERD (acid reflux).", opts) == "gastroesophageal reflux disease"
    assert chosen_option("Please see a doctor.", opts) is None
    assert option_exact_match("Most likely impetigo.", "impetigo", opts)
    assert not option_exact_match("Most likely impetigo.", "psoriasis", opts)


def test_parse_json_list():
    assert parse_json_list('<think>x</think> ["Rachel", "H&R"]') == ["Rachel", "H&R"]
    assert parse_json_list("nothing") == []
    assert parse_json_list('["a", "b"') == []
    assert parse_json_list('[{"name": "Mahsa zandiyeh"}, {"organization": "Indian Embassy"}]') == [
        "Mahsa zandiyeh", "Indian Embassy"]
    assert parse_json_list('{"names of people": ["Rachel Zheng"], "companies": ["H&R"]}') == ["Rachel Zheng", "H&R"]
    assert parse_json_list("[]") == []


def test_strip_think():
    assert strip_think("<think>\n\n</think>\n\nHello") == "Hello"


def _ctx(policy, tmp_path, edge, cloud, spans, enabled=True):
    guard = EgressGuard(policy, enabled=enabled)
    return Context(edge=edge, cloud=GuardedCloud(cloud, guard, tmp_path / "out.jsonl"),
                   labeler=StaticLabeler(spans), policy=policy)


EX = Example("e1", "Write an email from Rachel Zheng to Johnny Bay about the hotel.", "ref")
SP = [Span("Rachel Zheng", Label.LOCAL_ONLY, "PERSON"), Span("Johnny Bay", Label.LOCAL_ONLY, "PERSON")]


def test_redact_send_masks_and_restores(policy, tmp_path):
    cloud = FakeCloud(reply=lambda m: "Dear [PERSON_2], ... Best, [PERSON_1]")
    ctx = _ctx(policy, tmp_path, FakeEdge(), cloud, SP)
    tr = Trace("e1")
    ans = RedactSend()(EX, ctx, tr)
    sent = cloud.calls[0][0]["content"]
    assert "Rachel" not in sent and "Johnny" not in sent and "[PERSON_1]" in sent
    assert ans == "Dear Johnny Bay, ... Best, Rachel Zheng"
    c = request_cost(tr, 1.0)
    assert c["cloud_calls"] == 1 and c["cloud_in_tokens"] == 20 and c["edge_calls"] == 0


def test_redact_send_falls_back_when_guard_denies(policy, tmp_path):
    ctx = _ctx(policy, tmp_path, FakeEdge(["local email"]), FakeCloud(), SP)
    ctx.labeler = StaticLabeler(SP[:1])
    tr = Trace("e1")

    def send_with_all(messages, spans, **kw):
        return GuardedCloud.send(ctx.cloud, messages, SP, **kw)
    ctx.cloud.send = send_with_all
    assert RedactSend()(EX, ctx, tr) == "local email"
    assert tr.guard_denied and tr.notes["fallback"] == "edge_only"


def test_automix_verdict_parsing():
    from gapdeleg.baselines import verdict_correct
    assert verdict_correct("Fine.\nVerdict: Correct")
    assert not verdict_correct("It misses the date, so it is not correct.\nVerdict: Incorrect")
    assert not verdict_correct("<think>correct?</think>Verdict: **Incorrect**")
    assert verdict_correct("Correct") and not verdict_correct("") and not verdict_correct("unsure")


def test_automix_keeps_a_verified_edge_answer(policy, tmp_path):
    from gapdeleg.baselines import AutoMixGated
    edge = FakeEdge(["edge draft"] + ["Good.\nVerdict: Correct"] * 4)
    cloud = FakeCloud()
    tr = Trace("e1")
    assert AutoMixGated(k=4)(EX, _ctx(policy, tmp_path, edge, cloud, SP), tr) == "edge draft"
    assert cloud.calls == [] and len(edge.calls) == 5 and tr.notes["verification_score"] == 1.0
    assert "edge draft" in edge.calls[1][-1]["content"]


def test_automix_escalates_masked_when_verification_fails(policy, tmp_path):
    from gapdeleg.baselines import AutoMixGated
    edge = FakeEdge(["edge draft", "Verdict: Correct"] + ["Verdict: Incorrect"] * 3)
    cloud = FakeCloud(reply=lambda m: "Dear [PERSON_2], ... Best, [PERSON_1]")
    tr = Trace("e1")
    ans = AutoMixGated(k=4, threshold=0.5)(EX, _ctx(policy, tmp_path, edge, cloud, SP), tr)
    sent = cloud.calls[0][0]["content"]
    assert "Rachel" not in sent and "[PERSON_1]" in sent
    assert ans == "Dear Johnny Bay, ... Best, Rachel Zheng" and tr.notes["verification_score"] == 0.25


def test_automix_and_cascade_keep_the_edge_answer_when_the_guard_denies(policy, tmp_path):
    from gapdeleg.baselines import AutoMixGated, Cascade
    for system, replies, lps in ((AutoMixGated(k=2), ["edge draft", "Verdict: Incorrect", "Verdict: Incorrect"], (-0.1,)),
                                 (Cascade(), ["edge draft"], (-3.0,))):
        ctx = _ctx(policy, tmp_path, FakeEdge(replies, logprobs=lps), FakeCloud(), SP)
        ctx.labeler = StaticLabeler(SP[:1])

        def send_with_all(messages, spans, ctx=ctx, **kw):
            return GuardedCloud.send(ctx.cloud, messages, SP, **kw)
        ctx.cloud.send = send_with_all
        tr = Trace("e1")
        assert system(EX, ctx, tr) == "edge draft" and tr.guard_denied


def test_cascade_escalates_only_when_unsure(policy, tmp_path):
    from gapdeleg.baselines import Cascade
    sure = FakeCloud()
    assert Cascade(0.75)(EX, _ctx(policy, tmp_path, FakeEdge(["edge draft"], logprobs=(-0.05,)), sure, SP),
                         Trace("e1")) == "edge draft"
    assert sure.calls == []
    unsure = FakeCloud(reply=lambda m: "Hi [PERSON_2]")
    tr = Trace("e1")
    assert Cascade(0.75)(EX, _ctx(policy, tmp_path, FakeEdge(["edge draft"], logprobs=(-2.0,)), unsure, SP),
                         tr) == "Hi Johnny Bay"
    assert "Johnny" not in unsure.calls[0][0]["content"] and tr.notes["confidence"] < 0.75


def test_edge_only_and_cloud_full(policy, tmp_path):
    ctx = _ctx(policy, tmp_path, FakeEdge(["<think></think>hi"]), FakeCloud(), [], enabled=False)
    assert EdgeOnly()(EX, ctx, Trace("e1")) == "hi"
    assert CloudFull()(EX, ctx, Trace("e1")) == "cloud answer"


GAPS = '{"established": [], "uncertain": [{"text": "hotel options", "may_leave": true}], "missing": []}'


def test_gap_aware_runs_end_to_end_on_fakes(policy, tmp_path):
    edge = FakeEdge(["edge draft by Rachel Zheng", GAPS, "final for Johnny Bay from Rachel Zheng"], logprobs=(-2.0,))
    cloud = FakeCloud(reply=lambda m: "cloud draft for [PERSON_2]" if len(cloud.calls) == 1 else "notes on [PERSON_2]")
    ctx = _ctx(policy, tmp_path, edge, cloud, SP)
    tr = Trace("e1")
    ans = GapAwareDelegation(Flags(threshold=0.5))(EX, ctx, tr)
    assert ans == "final for Johnny Bay from Rachel Zheng"
    assert [s for s, _ in tr.cloud] == ["cloud_r1", "cloud_r2"]
    assert all("Rachel" not in m[0]["content"] and "Johnny" not in m[0]["content"] for m in cloud.calls)
    assert cloud.calls[0][0]["content"].startswith("Placeholders like [PERSON_1]")
    assert edge.calls[0][0]["content"] == EX.query
    assert "cloud draft for [PERSON_2]" in edge.calls[1][0]["content"]
    assert "Johnny" not in edge.calls[1][0]["content"]
    rewrite = edge.calls[2][0]["content"]
    assert EX.query in rewrite and "notes on Johnny Bay" in rewrite and "cloud draft for Johnny Bay" in rewrite
    assert tr.notes["answer_before_gaps"] == "cloud draft for Johnny Bay" and tr.notes["confidence"] < 0.5


def test_gap_aware_keeps_a_confident_edge_draft_and_masks_it_for_the_gap_report(policy, tmp_path):
    edge = FakeEdge(["Dear Johnny Bay, the hotel ...", '{"established": ["tone"]}'], logprobs=(-0.01,))
    cloud = FakeCloud()
    tr = Trace("e1")
    ans = GapAwareDelegation(Flags(threshold=0.5))(EX, _ctx(policy, tmp_path, edge, cloud, SP), tr)
    assert ans == "Dear Johnny Bay, the hotel ..." and cloud.calls == []
    assert "Dear [PERSON_2], the hotel" in edge.calls[1][0]["content"]
    assert tr.notes["gaps_r0"] == [] and tr.notes["gap_report_r0_parsed"]


def test_gap_aware_keeps_the_edge_draft_when_the_cloud_answer_is_empty(policy, tmp_path):
    edge = FakeEdge(["edge draft", "not json"], logprobs=(-3.0,))
    tr = Trace("e1")
    ans = GapAwareDelegation(Flags(threshold=0.5))(EX, _ctx(policy, tmp_path, edge, FakeCloud(reply=""), SP), tr)
    assert ans == "edge draft" and tr.notes["fallback_r1"] == "empty cloud answer"
    assert tr.notes["gap_report_r0_parsed"] is False and tr.notes["gaps_r0"] == []


def test_gap_aware_hides_what_the_guard_finds_and_asks_again(policy, tmp_path):
    ex = Example("e2", "Rachel Zheng has key d2J3BgtYdB2. Write her a reset note.", "ref")
    edge = FakeEdge(["edge draft"], logprobs=(-3.0,))
    cloud = FakeCloud(reply="Note for [PERSON_1]: your key [PRIVATE_1] was reset.")
    ctx = _ctx(policy, tmp_path, edge, cloud, SP[:1])
    ctx.cloud.guard.request_check = True
    tr = Trace("e2")
    ans = GapAwareDelegation(Flags(threshold=0.5, gap_loop=False))(ex, ctx, tr)
    assert len(cloud.calls) == 1 and "d2J3BgtYdB2" not in cloud.calls[0][0]["content"]
    assert "[PRIVATE_1]" in cloud.calls[0][0]["content"]
    assert [s for s, _ in tr.cloud] == ["cloud_r1_remasked"] and tr.notes["remasked"] == ["cloud_r1"]
    assert tr.guard_denied[0]["step"] == "cloud_r1"
    assert ans == "Note for Rachel Zheng: your key d2J3BgtYdB2 was reset."


def test_gap_aware_sends_nothing_when_hiding_cannot_fix_the_message(policy, tmp_path):
    edge = FakeEdge(["edge draft"], logprobs=(-3.0,))
    cloud = FakeCloud()
    ctx = _ctx(policy, tmp_path, edge, cloud, SP)
    ctx.cloud.guard.llm_check = lambda t, s, r: []
    ctx.cloud.guard.request_check = True
    ctx.cloud.guard.detector = type("D", (), {"detect": lambda self, t: [Span("hotel", Label.LOCAL_ONLY,
                                                                              "PERSON", score=0.99)]})()
    tr = Trace("e1")
    ans = GapAwareDelegation(Flags(threshold=0.5, gap_loop=False))(EX, ctx, tr)
    assert ans == "edge draft" and cloud.calls == [] and len(tr.guard_denied) == 2
    assert tr.notes["fallback_r1"] == "guard denied"


def test_gap_aware_rewrite_falls_back_and_masked_mode(policy, tmp_path):
    edge = FakeEdge(["edge draft", GAPS, ""], logprobs=(-3.0,))
    tr = Trace("e1")
    ans = GapAwareDelegation(Flags(threshold=0.5))(EX, _ctx(policy, tmp_path, edge, FakeCloud(reply="cloud text"), SP), tr)
    assert ans == "cloud text" and tr.notes["rewrite_r0_fallback"] == "draft kept"
    edge = FakeEdge(["edge draft", GAPS, "final for [PERSON_2]"], logprobs=(-3.0,))
    ans = GapAwareDelegation(Flags(threshold=0.5, rewrite="masked"))(
        EX, _ctx(policy, tmp_path, edge, FakeCloud(reply="text for [PERSON_2]"), SP), Trace("e1"))
    assert ans == "final for Johnny Bay" and "Johnny" not in edge.calls[2][0]["content"]
    with pytest.raises(ValueError):
        GapAwareDelegation(Flags(rewrite="nope"))


def test_guard_llm_check_calls_count_as_edge_steps(policy, tmp_path):
    from gapdeleg.privacy.guard import EdgeLeakCheck
    edge = FakeEdge(["edge draft", "1: no\n2: no"], logprobs=(-3.0,))
    ctx = _ctx(policy, tmp_path, edge, FakeCloud(), SP[:1])
    ctx.cloud.guard.llm_check = EdgeLeakCheck(edge)
    tr = Trace("e1")
    GapAwareDelegation(Flags(threshold=0.5, gap_loop=False))(EX, ctx, tr)
    assert [s for s, _ in tr.edge] == ["edge_attempt", "guard_cloud_r1"]
    assert ctx.cloud.guard.llm_check.last["candidates"] == ["Johnny", "Bay"]


def test_method_masks_values_next_to_cjk_text():
    from gapdeleg.split.template import mask
    sp = [Span("koushik", Label.LOCAL_ONLY, "PERSON")]
    assert mask("发邮件给koushik", sp, cjk_words=True).masked == "发邮件给[PERSON_1]"
    assert mask("发邮件给koushik", sp).masked == "发邮件给koushik"
    assert mask("koushik", sp, cjk_words=True).hide("写给koushik") == "写给[PERSON_1]"


def test_categorized_labeler_and_patterns(policy):
    from gapdeleg.privacy.labeler import CategorizedEdgeLabeler, PatternLabeler, parse_json_items
    assert parse_json_items('[{"text": "Erin", "category": "person"}, "Andy"]') == [("Erin", "PERSON"), ("Andy", "")]
    assert parse_json_items('{"names of people": ["Erin"], "PERSON": [{"text": "Andy"}]}') == \
        [("Erin", "NAMES_OF_PEOPLE"), ("Andy", "PERSON")]
    assert parse_json_items('[{"text": "Erin", "category": "PERSON"}, {"text": "Andy",]') == \
        [("Erin", ""), ("Andy", "")]
    reply = ('<think></think>[{"text": "Erin", "category": "PERSON"}, {"text": "Acme Corp", "category": '
             '"ORGANIZATION"}, {"text": "I feel shy", "category": "PERSONAL_FACT"}, {"text": "Bob", "category": '
             '"PERSON"}, {"text": "night shift", "category": "JOB"}]')
    spans = CategorizedEdgeLabeler(FakeEdge([reply]), policy).label("Erin at Acme Corp works the night shift. I feel shy.")
    got = {s.text: (s.label, s.category) for s in spans}
    assert got == {"Erin": (Label.LOCAL_ONLY, "PERSON"), "Acme Corp": (Label.ABSTRACTABLE, "ORGANIZATION"),
                   "I feel shy": (Label.LOCAL_ONLY, "PERSONAL_FACT"), "night shift": (Label.LOCAL_ONLY, "PRIVATE")}
    pat = {s.text: s.category for s in PatternLabeler(policy).label(
        "id cc6a2651-f67d-4e9a-980c-63d72a62f2d3, key d2J3BgtYdB2, @Xtension11, call 514 555 0199, rz@x.org")}
    assert pat == {"cc6a2651-f67d-4e9a-980c-63d72a62f2d3": "ID_NUMBER", "d2J3BgtYdB2": "ID_NUMBER",
                   "@Xtension11": "ID_NUMBER", "514 555 0199": "ID_NUMBER", "rz@x.org": "EMAIL_ADDRESS"}


def test_personal_phrase_labeler_copies_phrases_from_the_request():
    from gapdeleg.experiment import build_labeler
    from gapdeleg.privacy.labeler import PersonalPhraseLabeler, locate
    text = "hi, I am trying to reduce my body weight. Rhonda was not aware of the unpaid premiums."
    assert locate("UNPAID PREMIUMS", text) == "unpaid premiums"
    assert locate("I am trying to reduce my body wieght", text) == "I am trying to reduce my body weight"
    assert locate("I want to lose weight fast", text) is None
    assert locate("weight loss", text) is None
    reply = '<think></think>["I am trying to reduce my bodyweight", "unpaid premiums", "she is very rich", "x"]'
    edge = FakeEdge([reply])
    spans = PersonalPhraseLabeler(edge).label(text)
    assert {s.text: (s.label, s.category) for s in spans} == {
        "I am trying to reduce my body weight": (Label.LOCAL_ONLY, "PERSONAL_FACT"),
        "unpaid premiums": (Label.LOCAL_ONLY, "PERSONAL_FACT")}
    assert text in edge.calls[0][0]["content"]
    assert len(PersonalPhraseLabeler(FakeEdge(["[]"])).label(text)) == 0
    names = [type(p).__name__ for p in build_labeler(["edge_phrases"], None, FakeEdge([]), None).labelers]
    assert names == ["PersonalPhraseLabeler"]


def test_code_patterns_skip_common_technical_words():
    from gapdeleg.privacy.detectors import code_values
    assert code_values("sha256 in Python3 on COVID19 data from 2026, 4K video, x86_64, [PERSON_1], 12345") == []
    assert code_values("color: #2A332F; bgcolor=\"#7754b3\"; yaml转换为k8s") == []


def test_cost_aggregate():
    a = aggregate([{"x": 1, "y": 2.0}, {"x": 3, "y": 4.0}])
    assert a["x"] == {"sum": 4, "mean": 2, "median": 2.0}


def test_generation_roundtrip():
    g = Generation("t", [-0.1], ["t"], Usage(1, 2), 0.5, "m")
    assert Generation.from_dict(g.to_dict()) == g
