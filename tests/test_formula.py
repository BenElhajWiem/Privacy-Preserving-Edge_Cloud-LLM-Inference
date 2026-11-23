import pytest

from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace
from gapdeleg.pipeline.egress import GuardedCloud
from gapdeleg.privacy.guard import EgressGuard
from gapdeleg.privacy.types import Label, Span
from gapdeleg.formula.formula import (FormulaFlags, FormulaProgram, build_prompt, clean_mask, extract_code, make,
                                     name_runs, number_type)
from gapdeleg.formula.formula_sandbox import RunError, StaticError, compile_program

from conftest import FakeCloud, FakeEdge


class StaticLabeler:
    def __init__(self, spans):
        self.spans = spans

    def label(self, text):
        return self.spans


def _ctx(policy, tmp_path, edge, cloud, spans):
    return Context(edge=edge, cloud=GuardedCloud(cloud, EgressGuard(policy), tmp_path / "out.jsonl"),
                   labeler=StaticLabeler(spans), policy=policy)


Q = ("Hi, I'm Imani Okafor. I withdrew $25. Tickets are $8 per person, and it's the two of us. My brother Theo "
     "pays twice as much for snacks as one ticket. How much change do we get if I also pay for his snacks?\n\n"
     'Please finish your reply with a last line that starts with "Answer:" followed by the final number.')
EX = Example("f1", Q, "-7", family="B")
SPANS = [Span("Imani Okafor", Label.LOCAL_ONLY, "PERSON"), Span("My brother Theo", Label.LOCAL_ONLY, "PERSON"),
         Span("How much change do we get", Label.LOCAL_ONLY, "PERSONAL_FACT")]
PROGRAM = ("Subtract the tickets and the snacks from the cash.\n```python\nimport math\n\n"
           "def solve(NUM_1, NUM_2, NUM_3, RATIO_1):\n    tickets = NUM_2 * NUM_3\n"
           "    snacks = RATIO_1 * NUM_2\n    return NUM_1 - tickets - snacks\n```\nAnswer: unknown")


M = {"[NUM_1]": "25", "[NUM_2]": "8", "[NUM_3]": "two", "[RATIO_1]": "twice", "[PERSON_1]": "Imani Okafor"}
V = {"NUM_1": 25.0, "NUM_2": 8.0, "NUM_3": 2.0, "RATIO_1": 2.0}


def test_sandbox_runs_functions_loops_and_math():
    assert compile_program(PROGRAM.split("```python\n")[1].split("```")[0], M).run(V) == -7
    loop = "def solve(NUM_1, NUM_2):\n    weeks = 0\n    while NUM_1 > NUM_2:\n        NUM_1 = NUM_1 / 2\n" \
           "        weeks += 1\n    return weeks"
    assert compile_program(loop, M).run({"NUM_1": 48.0, "NUM_2": 6.0}) == 3
    assert compile_program("from math import ceil\ndef solve(NUM_1):\n    return ceil(NUM_1 * 0.1 * 30)",
                           M).run({"NUM_1": 1.0}) == 3
    assert compile_program("def solve(NUM_1):\n    return round(NUM_1)", M).run({"NUM_1": 2.5}) == 3
    assert compile_program("def solve(NUM_1, NUM_2):\n    return max(NUM_1, NUM_2) // 3 + NUM_1 % 3 + 2 ** 3",
                           M).run(V) == 8 + 1 + 8
    assert compile_program("total = 0\nfor i in range(1, 4):\n    total += [NUM_1] * i\nanswer = total",
                           M).run(V) == 150
    other = "def cost(price, n):\n    return price * n\n\nresult = cost(NUM_2, NUM_3)\nprint(cost(1, 2))"
    p = compile_program(other, M)
    assert p.mode == "call" and p.run(V) == 16
    assert compile_program("def solve(NUM_1, minutes=60):\n    return NUM_1 * minutes", M).run(V) == 1500
    assert compile_program("def solve(NUM_1, NUM_9):\n    return NUM_1", M).run(V) == 25
    with pytest.raises(StaticError, match="NUM_9"):
        compile_program("def solve(NUM_1, NUM_9):\n    return NUM_1 + NUM_9", M)


@pytest.mark.parametrize("code, msg", [
    ("import os\ndef solve(NUM_1):\n    return NUM_1", "only `import math`"),
    ("def solve(NUM_1):\n    return __import__('os')", "strings"),
    ("def solve(NUM_1):\n    return eval(NUM_1)", "eval()"),
    ("def solve(NUM_1):\n    return NUM_1.real", "attribute"),
    ("def solve(NUM_1):\n    return [NUM_1][0]", "indexing"),
    ("def solve(NUM_1):\n    return sum(x for x in range(3))", "generator"),
    ("def solve(NUM_1):\n    return lambda: 1", "lambda"),
    ("def solve(NUM_1):\n    return unknown + 1", "not defined"),
    ("def solve(PERSON_1):\n    return PERSON_1 + 1", "not a number"),
    ("def solve(price):\n    return price * 2", "not a placeholder"),
    ("def solve(NUM_1):\n    return NUM_1 +", "syntax error"),
    ("", "empty"),
])
def test_sandbox_rejects_before_running(code, msg):
    with pytest.raises(StaticError, match=msg):
        compile_program(code, M)


def test_sandbox_value_errors_are_run_errors():
    for code, values in [("def solve(NUM_1):\n    return 1 / (NUM_1 - 25)", V),
                         ("def solve(NUM_1):\n    while True:\n        pass", V),
                         ("def solve(NUM_1):\n    return NUM_1 ** 100", V),
                         ("def solve(NUM_1):\n    return NUM_1 > 3", V),
                         ("def solve(NUM_1):\n    x = 1", V),
                         ("def solve(NUM_1):\n    for i in range(NUM_1):\n        pass\n    return 1", {"NUM_1": 2.5}),
                         ("def solve(NUM_1):\n    return solve(NUM_1)", V)]:
        with pytest.raises(RunError):
            compile_program(code, M).run(values)


def test_name_runs_keep_names_only():
    assert name_runs("Their oldest sister Marcia") == ["Marcia"]
    assert name_runs("How old was Greg?") == ["Greg"]
    assert name_runs("from my Summit Community Bank account") == ["Summit Community Bank"]
    assert name_runs("Gerald's fourth offense") == ["Gerald"]
    assert name_runs("Snap, Crackle, and Pop") == ["Snap", "Crackle", "Pop"]
    assert name_runs("in total") == [] and name_runs("On Sunday") == []
    assert number_type("twice") == number_type("three-quarters") == number_type("1/3") == "RATIO"
    assert number_type("two") == number_type("A dozen") == number_type("1,250") == "NUM"


def test_clean_mask_hides_every_number_and_reduces_labeled_phrases_to_names():
    tpl, guard_spans = clean_mask(Q + " Account 2480-1562-7160.", SPANS + [
        Span("2480-1562-7160", Label.LOCAL_ONLY, "ACCOUNT_NUMBER")])
    m = tpl.masked
    assert "$[NUM_1]" in m and "the [NUM_3] of us" in m and "[RATIO_1] as much" in m
    assert "My brother [PERSON_2] pays" in m and "How much change do we get" in m
    assert "[ACCOUNT_NUMBER_1]" in m and not any(v in m for v in ("25", "8", "two", "twice", "Theo", "Imani"))
    assert {"25", "8", "two", "twice", "Theo"} <= {s.text for s in guard_spans}


def test_extract_code_prefers_the_function_block():
    reply = "Plan.\n```python\nprint(solve(1))\n```\nThen\n```python\ndef solve(NUM_1):\n    return NUM_1\n```\nmore"
    assert extract_code(reply) == ("def solve(NUM_1):\n    return NUM_1\n", "block")
    assert extract_code("Plan\n```python\ndef solve(NUM_1):\n    return")[1] == "unclosed block"
    assert extract_code("x = [NUM_1] * 2\nanswer = x") == ("x = [NUM_1] * 2\nanswer = x", "lines")


def test_prompts_hold_no_number_the_guard_could_match():
    import re
    from gapdeleg.pipeline.method_request import NUM_RE
    from gapdeleg.formula.formula import REPAIR
    for prompt in ("py", "py_ex"):
        for mode in ("v0", "clean"):
            text = re.sub(r"\[?[A-Z]+_(?:\d+|k)\]?", "", build_prompt("", prompt, mode) + REPAIR)
            assert NUM_RE.findall(text) == [], (prompt, mode)


def test_prompts_example_and_types():
    p = build_prompt("PROBLEM", "py", "v0")
    assert "PROBLEM" in p and "Example" not in p and "[RATIO_k]" not in p
    assert "[RATIO_k]" in build_prompt("PROBLEM", "py", "clean")
    ex = build_prompt("PROBLEM", "py_ex", "v0")
    assert "Example" in ex and "RATIO" not in ex and "NUM_3 * NUM_4" in ex
    assert "RATIO_1 = " not in build_prompt("P", "py_ex", "clean") and "def solve" in build_prompt("P", "py_ex",
                                                                                                    "clean")
    with pytest.raises(ValueError):
        make(prompt="nope")


def test_program_runs_on_the_device_and_only_masked_text_leaves(policy, tmp_path):
    edge = FakeEdge(["edge draft\nAnswer: 1"], logprobs=(-3.0,))
    cloud = FakeCloud(reply=PROGRAM)
    tr = Trace("f1")
    ans = make(mask="clean")(EX, _ctx(policy, tmp_path, edge, cloud, SPANS), tr)
    sent = cloud.calls[0][0]["content"]
    assert all(v not in sent for v in ("25", "$8", "two of", "twice", "Imani", "Theo")) and "def solve" in sent
    assert ans.endswith("Answer: -7") and "unknown" not in ans and "NUM_1 = 25" in ans
    n = tr.notes
    assert n["route"] == "program" and n["ran"] and n["value"] == -7 and not n["repaired"]
    assert n["bindings"]["RATIO_1"] == "twice" and n["draft_number"] == "1" and len(edge.calls) == 1


def test_static_error_goes_back_once_then_the_fixed_program_runs(policy, tmp_path):
    replies = iter(["```python\ndef solve(NUM_1, NUM_2):\n    return NUM_1.real\n```", PROGRAM])
    cloud = FakeCloud(reply=lambda messages: next(replies))
    tr = Trace("f1")
    ans = make(mask="clean")(EX, _ctx(policy, tmp_path, FakeEdge(["d"], logprobs=(-3.0,)), cloud, SPANS), tr)
    assert ans.endswith("Answer: -7") and tr.notes["repaired"] and len(cloud.calls) == 2
    first, (prompt, earlier, repair) = cloud.calls[0], cloud.calls[1]
    assert prompt == first[0] and earlier["role"] == "assistant" and "NUM_1.real" in earlier["content"]
    assert "could not run: the attribute .real" in repair["content"]
    assert not any(v in m["content"] for m in (prompt, repair) for v in ("25", "$8", "two of", "twice", "Theo"))
    assert [s for s, _ in tr.cloud] == ["cloud_method", "cloud_repair"] and "remasked" not in tr.notes


def test_run_error_is_not_sent_back_and_the_edge_applies_the_method(policy, tmp_path):
    cloud = FakeCloud(reply="```python\ndef solve(NUM_1, NUM_2):\n    return NUM_1 / (NUM_2 - 8)\n```")
    edge = FakeEdge(["draft\nAnswer: 1", "worked\nAnswer: -7"], logprobs=(-3.0,))
    tr = Trace("f1")
    ans = make(mask="v0")(EX, _ctx(policy, tmp_path, edge, cloud, SPANS), tr)
    assert ans == "worked\nAnswer: -7" and len(cloud.calls) == 1
    assert tr.notes["attempts"][0]["run_error"] == "division by zero" and not tr.notes["ran"]
    assert tr.notes["fallback"].startswith("edge applies the method")
    apply = edge.calls[1][0]["content"]
    assert Q in apply and "NUM_2 is '8'" in apply
    tr2 = Trace("f1")
    assert make(fallback="draft")(EX, _ctx(policy, tmp_path, FakeEdge(["draft"], logprobs=(-3.0,)), cloud, SPANS),
                                  tr2) == "draft"
    assert tr2.notes["fallback"].startswith("edge draft (program did not run")


def test_gate_keeps_a_confident_draft_and_empty_reply_keeps_the_draft(policy, tmp_path):
    cloud = FakeCloud(reply=PROGRAM)
    tr = Trace("f1")
    sys = FormulaProgram(FormulaFlags(gate=0.5))
    assert sys(EX, _ctx(policy, tmp_path, FakeEdge(["sure"], logprobs=(-0.01,)), cloud, SPANS), tr) == "sure"
    assert cloud.calls == [] and tr.notes["route"] == "draft (confident)"
    tr = Trace("f1")
    assert make()(EX, _ctx(policy, tmp_path, FakeEdge(["d"], logprobs=(-3.0,)), FakeCloud(reply=""), SPANS),
                  tr) == "d"
    assert tr.notes["fallback"] == "edge draft (empty cloud reply)"
