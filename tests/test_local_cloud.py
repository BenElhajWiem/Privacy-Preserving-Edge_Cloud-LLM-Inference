import pytest

from gapdeleg.backends import UnguardedEgressError, make_cloud
from gapdeleg.backends.base import Generation, Usage
from gapdeleg.backends.ollama_cloud import OllamaCloud
from gapdeleg.baselines import EdgeOnly, make_system


def test_make_cloud_builds_the_local_cloud(tmp_path):
    c = make_cloud({"backend": "ollama", "model": "qwen3:32b-q4_K_M", "host": "http://127.0.0.1:1"},
                   cache_dir=tmp_path)
    assert isinstance(c, OllamaCloud) and c.name == "qwen3:32b-q4_K_M" and c.cache_dir == tmp_path


def test_local_cloud_needs_a_ticket_and_passes_only_ollama_parameters(tmp_path, monkeypatch):
    c = OllamaCloud("m", host="http://127.0.0.1:1", cache_dir=tmp_path)
    with pytest.raises(UnguardedEgressError):
        c.generate([{"role": "user", "content": "hi"}], ticket=None)
    seen = {}

    def fake(messages, **kw):
        seen.update(kw)
        return Generation(text="ok", usage=Usage(3, 1))
    monkeypatch.setattr(c._model, "generate", fake)
    assert c._generate([{"role": "user", "content": "hi"}], max_tokens=7, effort="low", format="json").text == "ok"
    assert seen == {"max_tokens": 7, "temperature": 0.0, "logprobs": False, "format": "json"}


def test_make_system_takes_a_module_factory():
    assert isinstance(make_system("gapdeleg.baselines:EdgeOnly"), EdgeOnly)
