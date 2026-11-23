import io
import json

import pytest

from gapdeleg.data.loaders import EXTERNAL, load_gsm8k, load_prism, load_pupa
from gapdeleg.privacy.policy import Policy
from gapdeleg.privacy.types import Label

needs_external = pytest.mark.skipif(not (EXTERNAL / "PAPILLON").exists(), reason="external/ not cloned")


@needs_external
def test_pupa():
    tnb = load_pupa("tnb")
    assert len(tnb) == 237
    ex = tnb[0]
    assert ex.private_spans and all(s.label == Label.LOCAL_ONLY for s in ex.private_spans)
    assert any(s.start >= 0 and ex.query[s.start:s.end].lower() == s.text.lower() for s in ex.private_spans)
    assert len(load_pupa("new", limit=5)) == 5


@needs_external
def test_prism_and_gsm8k():
    assert len(load_prism()) == 160
    assert {e.meta["domain"] for e in load_prism()} == {"Medical", "Tourism", "Bank", "Common"}
    g = load_gsm8k(limit=1)[0]
    assert g.reference == "18" and g.family == "B"


def test_policy_defaults():
    p = Policy.load()
    assert p.label_for("person") == Label.LOCAL_ONLY
    assert p.label_for("ORGANIZATION") == Label.ABSTRACTABLE
    assert p.label_for("SOMETHING_NEW") == Label.LOCAL_ONLY


def test_ollama_backend_parses_response(monkeypatch):
    from gapdeleg.backends import ollama_backend as ob
    sent = {}
    reply = {"message": {"content": "42"}, "prompt_eval_count": 12, "eval_count": 2, "eval_duration": 2e7,
             "load_duration": 0, "logprobs": [{"token": "4", "logprob": -0.1}, {"token": "2", "logprob": -0.3}]}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr(ob.urllib.request, "urlopen", fake_urlopen)
    g = ob.OllamaBackend("qwen3:1.7b").generate([{"role": "user", "content": "q"}], max_tokens=8)
    assert g.text == "42" and g.token_logprobs == [-0.1, -0.3] and g.tokens == ["4", "2"]
    assert (g.usage.input_tokens, g.usage.output_tokens) == (12, 2)
    assert g.meta["eval_tok_per_s"] == pytest.approx(100)
    assert sent["body"]["think"] is False and sent["body"]["logprobs"] is True
    assert sent["body"]["options"]["num_predict"] == 8
    assert "format" not in sent["body"]
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    ob.OllamaBackend("qwen3:1.7b").generate([{"role": "user", "content": "q"}], format=schema)
    assert sent["body"]["format"] == schema


def test_anthropic_backend_requires_own_key(monkeypatch, policy):
    from gapdeleg.backends.anthropic_backend import AnthropicBackend
    from gapdeleg.privacy.guard import EgressGuard
    monkeypatch.delenv("GAPDELEG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-used")
    m = [{"role": "user", "content": "q"}]
    with pytest.raises(RuntimeError, match="GAPDELEG_ANTHROPIC_API_KEY"):
        AnthropicBackend().generate(m, ticket=EgressGuard(policy).check(m, []).ticket)
