import pytest

from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace
from gapdeleg.pipeline.egress import GuardedCloud
from gapdeleg.pipeline.method_request import (MethodFlags, MethodRequest, extract_program, format_number,
                                              formula_mask, parse_options, run_program, to_number)
from gapdeleg.privacy.guard import EgressGuard
from gapdeleg.privacy.types import Label, Span
from gapdeleg.split.template import mask

from conftest import FakeCloud, FakeEdge


class StaticLabeler:
    def __init__(self, spans):
        self.spans, self.seen = spans, []

    def label(self, text):
        self.seen.append(text)
        return self.spans


def _ctx(policy, tmp_path, edge, cloud, spans):
    return Context(edge=edge, cloud=GuardedCloud(cloud, EgressGuard(policy), tmp_path / "out.jsonl"),
                   labeler=StaticLabeler(spans), policy=policy)


FIN = Example("f1", "Hi, I'm Imani Okafor. I withdrew $25. Tickets are $8 per person, and it's two of us. "
                    "How much change do we get?\n\nPlease finish your reply with a last line that starts with "
                    '"Answer:" followed by the final number.', "9", family="B")
FIN_SPANS = [Span("Imani Okafor", Label.LOCAL_ONLY, "PERSON"), Span("$25", Label.LOCAL_ONLY, "MONEY"),
             Span("$8", Label.LOCAL_ONLY, "MONEY"), Span("two", Label.LOCAL_ONLY, "AMOUNT")]
METHOD = ("Subtract the cost of the tickets from the cash.\n```\ntickets = [NUM_2] * [NUM_3]\n"
          "answer = [NUM_1] - tickets\n```")

OPTIONS = "allergy, bronchial asthma, common cold, pneumonia"


def _med(name, symptoms):
    return Example(name, f"Hi, I'm {name}. I've had {symptoms}. Which of these is the most likely diagnosis: "
                         f"{OPTIONS}? Please name one.", "bronchial asthma", family="B")


def test_to_number_reads_digits_words_and_fractions():
    cases = {"$1,250.50": 1250.5, "15%": 15, "3/4": 0.75, "three-quarters": 0.75, "two dozen": 24, "5 years old": 5,
             "twenty-one": 21, "one thousand two hundred": 1200, "half": 0.5, "twice": 2, "Imani": None,
             "I buy 7 apples a week for the team at $2 each": None, "between 3 and 5": None}
    for text, want in cases.items():
        assert to_number(text) == (pytest.approx(want) if want is not None else None), text


def test_run_program_uses_the_hidden_values_and_nothing_else():
    tpl = mask("I have $25 and two tickets at $8. I am Ann Lee.", [
        Span("$25", Label.LOCAL_ONLY, "MONEY"), Span("two", Label.LOCAL_ONLY, "AMOUNT"),
        Span("$8", Label.LOCAL_ONLY, "MONEY"), Span("Ann Lee", Label.LOCAL_ONLY, "PERSON")])
    assert run_program(["t = [MONEY_2] * AMOUNT_1", "Answer = [MONEY_1] - t"], tpl) == 9
    assert run_program(["x = (1,000 + $5) / 2 × 3"], tpl) == pytest.approx(1507.5)
    for bad in (["a = __import__('os')"], ["a = [PERSON_1] + 1"], ["a = MONEY_1 ** 2"], ["a = b + 1"], []):
        with pytest.raises(ValueError):
            run_program(bad, tpl)
    with pytest.raises(ZeroDivisionError):
        run_program(["a = [MONEY_1] / 0"], tpl)


def test_extract_program_format_and_options():
    assert extract_program(METHOD) == ["tickets = [NUM_2] * [NUM_3]", "answer = [NUM_1] - tickets"]
    assert extract_program("Steps:\nx = 2 * 3  # double\nThen x == 6\nanswer = x") == ["x = 2 * 3", "answer = x"]
    assert (format_number(25.0), format_number(0.5), format_number(2 / 3)) == ("25", "0.5", "0.666667")
    assert parse_options(_med("Ann", "a cough").query) == OPTIONS.split(", ")
    assert parse_options(FIN.query) == []


def test_formula_mask_hides_every_number_but_keeps_units():
    q = ("ID WA8707974, account 3583-8830-8779, I'm Ann Lee: a fifth of an hour each week, $25, 15% and two dozen. "
         "How many minutes?")
    spans = [Span("WA8707974", Label.LOCAL_ONLY, "ID_NUMBER"), Span("3583-8830-8779", Label.LOCAL_ONLY, "ACCOUNT_NUMBER"),
             Span("each week", Label.ABSTRACTABLE, "DATE_TIME"), Span("$25", Label.ABSTRACTABLE, "MONEY"),
             Span("Ann Lee", Label.LOCAL_ONLY, "PERSON"), Span("minutes", Label.ABSTRACTABLE, "MONEY")]
    tpl, guard_spans = formula_mask(q, spans)
    assert tpl.masked == ("ID [ID_NUMBER_1], account [ACCOUNT_NUMBER_1], I'm [PERSON_1]: [NUM_1] of an hour each "
                          "week, $[NUM_2], [NUM_3]% and [NUM_4]. How many minutes?")
    assert run_program(["a = [NUM_1] * 60 + [NUM_2] + [NUM_3] + [NUM_4]"], tpl) == pytest.approx(76)
    assert {"25", "15", "two dozen"} <= {s.text for s in guard_spans}


def test_formula_masks_the_values_and_computes_on_the_device(policy, tmp_path):
    edge = FakeEdge(["edge draft\nAnswer: 17"], logprobs=(-3.0,))
    cloud = FakeCloud(reply=METHOD + "\nAnswer: unknown")
    tr = Trace("f1")
    ans = MethodRequest(MethodFlags(threshold=0.5))(FIN, _ctx(policy, tmp_path, edge, cloud, FIN_SPANS), tr)
    assert "unknown" not in ans
    sent = cloud.calls[0][0]["content"]
    assert all(v not in sent for v in ("25", "8", "two", "Imani")) and "$[NUM_1]" in sent
    assert ans.endswith("Answer: 9") and "8 * two" in ans
    assert tr.notes["kind"] == "formula" and tr.notes["value"] == 9 and len(edge.calls) == 1


def test_formula_falls_back_to_the_edge_model_when_the_program_cannot_run(policy, tmp_path):
    edge = FakeEdge(["edge draft", "worked out\nAnswer: 9"], logprobs=(-3.0,))
    cloud = FakeCloud(reply="Use the change.\n```\nanswer = [PERSON_1] - [NUM_2]\n```")
    tr = Trace("f1")
    ans = MethodRequest(MethodFlags(threshold=0.5))(FIN, _ctx(policy, tmp_path, edge, cloud, FIN_SPANS), tr)
    assert ans == "worked out\nAnswer: 9" and tr.notes["fallback"] == "edge applies the method"
    apply = edge.calls[1][0]["content"]
    assert FIN.query in apply and "Imani Okafor - 8" in apply


def test_confident_draft_or_denied_message_keeps_the_draft(policy, tmp_path):
    cloud = FakeCloud(reply=METHOD)
    ans = MethodRequest(MethodFlags(threshold=0.5))(
        FIN, _ctx(policy, tmp_path, FakeEdge(["sure\nAnswer: 9"], logprobs=(-0.01,)), cloud, FIN_SPANS), Trace("f1"))
    assert ans == "sure\nAnswer: 9" and cloud.calls == []
    ctx = _ctx(policy, tmp_path, FakeEdge(["unsure"], logprobs=(-3.0,)), cloud, FIN_SPANS[:1])
    ctx.cloud.guard.request_check = True
    ctx.cloud.guard.detector = type("D", (), {"detect": lambda self, t: [Span("Tickets", Label.LOCAL_ONLY,
                                                                              "PERSON", score=0.99)]})()
    tr = Trace("f1")
    assert MethodRequest(MethodFlags(threshold=0.5))(FIN, ctx, tr) == "unsure" and cloud.calls == []
    assert tr.notes["fallback"] == "guard denied" and len(tr.guard_denied) == 2


def test_checklist_sends_only_the_options_and_applies_them_on_the_device(policy, tmp_path):
    sent = []
    for ex in (_med("Ann Lee", "wheezing at night"), _med("Bo Chan", "a fever and chills")):
        spans = [Span(ex.id, Label.LOCAL_ONLY, "PERSON")]
        edge = FakeEdge(["draft", "Final answer: bronchial asthma"], logprobs=(-3.0,))
        cloud = FakeCloud(reply="bronchial asthma: wheezing, tight chest")
        ctx = _ctx(policy, tmp_path, edge, cloud, spans)
        tr = Trace(ex.id)
        assert MethodRequest(MethodFlags(threshold=0.5))(ex, ctx, tr) == "Final answer: bronchial asthma"
        sent.append(cloud.calls[0][0]["content"])
        assert OPTIONS in sent[-1] and ex.id not in sent[-1] and tr.notes["kind"] == "checklist"
        assert "Which of these" not in ctx.labeler.seen[0]
        apply = edge.calls[1][0]["content"]
        assert ex.query in apply and "wheezing, tight chest" in apply
    assert sent[0] == sent[1]
    with pytest.raises(ValueError):
        MethodRequest(MethodFlags(kind="nope"))
