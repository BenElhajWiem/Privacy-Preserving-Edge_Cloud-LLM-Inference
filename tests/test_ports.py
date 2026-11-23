import pytest

from gapdeleg.data.loaders import EXTERNAL
from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace
from gapdeleg.pipeline.egress import GuardedCloud
from gapdeleg.privacy.guard import EgressGuard

from conftest import FakeCloud, FakeEdge

needs_papillon = pytest.mark.skipif(not (EXTERNAL / "PAPILLON").exists(), reason="external/ not cloned")
needs_prism = pytest.mark.skipif(not (EXTERNAL / "PRISM").exists(), reason="external/ not cloned")


@needs_papillon
def test_papillon_loads_published_prompts_and_runs(tmp_path, policy):
    from gapdeleg.baselines.papillon import PapillonPort, load_instructions
    creator, aggregator = load_instructions("llama_32_3b")
    assert "privacy" in creator.lower() and aggregator
    edge = FakeEdge(["[[ ## reasoning ## ]]\nhide the name\n\n[[ ## createdPrompt ## ]]\nWrite a polite email to a "
                     "colleague about a hotel booking.\n\n[[ ## completed ## ]]",
                     "[[ ## finalOutput ## ]]\nDear Johnny, ...\n\n[[ ## completed ## ]]"])
    cloud = FakeCloud(reply="Here is a draft email.")
    port = PapillonPort("llama_32_3b")
    assert port.guard_disabled
    ctx = Context(edge=edge, cloud=GuardedCloud(cloud, EgressGuard(policy, enabled=False)), labeler=None,
                  policy=policy)
    tr = Trace("e")
    out = port(Example("e", "Rachel asks Johnny Bay to book a hotel. Write the email."), ctx, tr)
    assert out == "Dear Johnny, ..."
    assert cloud.calls[0][0]["content"].startswith("Write a polite email")
    assert "Rachel" in edge.calls[1][-1]["content"]
    assert "Here is a draft email." in edge.calls[1][-1]["content"]


@needs_papillon
def test_papillon_retries_in_json_and_sends_only_the_created_prompt(policy):
    from gapdeleg.baselines.papillon import PapillonPort
    edge = FakeEdge(['We are given a user query: "Hi I\'m Belal a GP doctor". Avoid the name Belal.',
                     '{"reasoning": "hide Belal", "createdPrompt": "A GP introduces themselves; reply politely."}',
                     "[[ ## finalOutput ## ]]\nok\n\n[[ ## completed ## ]]"])
    cloud = FakeCloud()
    ctx = Context(edge=edge, cloud=GuardedCloud(cloud, EgressGuard(policy, enabled=False)), labeler=None,
                  policy=policy)
    tr = Trace("e")
    assert PapillonPort(None)(Example("e", "Hi I'm Belal a GP doctor"), ctx, tr) == "ok"
    assert "JSON object" in edge.calls[1][-1]["content"]
    assert cloud.calls[0][0]["content"] == "A GP introduces themselves; reply politely."
    assert tr.notes["prompt_creator_format"] == "json" and tr.notes["info_aggregator_format"] == "chat"


@needs_papillon
def test_papillon_sends_nothing_when_both_formats_fail(policy):
    from gapdeleg.baselines.papillon import PapillonPort
    cloud = FakeCloud()
    ctx = Context(edge=FakeEdge(["We are given a user query about Belal ...", "still not JSON about Belal"]),
                  cloud=GuardedCloud(cloud, EgressGuard(policy, enabled=False)), labeler=None, policy=policy)
    tr = Trace("e")
    assert PapillonPort(None)(Example("e", "Hi I'm Belal a GP doctor"), ctx, tr) == ""
    assert cloud.calls == []
    assert tr.notes["failed"] == "prompt creator output unparseable"


def test_prism_turns_qwen3_thinking_off_in_raw_prompts():
    from gapdeleg.baselines.prism import QWEN3_NO_THINK, no_think_suffix
    assert no_think_suffix("qwen3:1.7b") == QWEN3_NO_THINK == no_think_suffix("qwen3:4b-q4_K_M")
    assert no_think_suffix("qwen3:4b-instruct-2507-q4_K_M") == "" == no_think_suffix("llama3.2:3b")


@needs_prism
def test_prism_edge_examples_parse_fix():
    from gapdeleg.baselines.prism import CODE, parse_edge_examples
    ex = parse_edge_examples((CODE / "few_shot_examples_edge.txt").read_text())
    assert {k: len(v) for k, v in ex.items()} == {"Tourism": 2, "Medical": 2, "Banking": 2, "Common": 0}
    assert all(e["response"] and not e["response"].startswith('"') for v in ex.values() for e in v)
    assert "Day 2" in ex["Tourism"][0]["response"]
