import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env"))


def test_ollama_logprobs():
    from gapdeleg.backends.ollama_backend import OllamaBackend
    b = OllamaBackend("qwen3:0.6b")
    assert b.describe()["digest"]
    g = b.generate([{"role": "user", "content": "What is 2+2? Number only."}], max_tokens=8)
    assert "4" in g.text and g.token_logprobs


def test_detector_finds_pii():
    from gapdeleg.privacy.detectors import Detector
    cats = {s.category for s in Detector().detect("Rachel Zheng, rachel@example.com, 212-555-0199")}
    assert {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"} <= cats


def test_anthropic_through_guard(policy):
    from envfile import load_env
    load_env()
    from gapdeleg.backends.anthropic_backend import AnthropicBackend
    from gapdeleg.privacy.guard import EgressGuard
    m = [{"role": "user", "content": "Reply with the single word: ready"}]
    g = AnthropicBackend("claude-haiku-5-5").generate(m, ticket=EgressGuard(policy).check(m, []).ticket, max_tokens=8)
    assert "ready" in g.text.lower()
