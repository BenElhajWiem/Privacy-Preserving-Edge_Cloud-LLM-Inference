from __future__ import annotations

from .base import CloudModel, Generation, Message
from .ollama_backend import DEFAULT_HOST, OllamaBackend


class OllamaCloud(CloudModel):
    def __init__(self, model: str, host: str = DEFAULT_HOST, think: bool = False, num_ctx: int = 8192,
                 seed: int | None = 0, timeout: float = 900, cache_dir=None):
        super().__init__(cache_dir)
        self.name = model
        self._model = OllamaBackend(model, host=host, think=think, num_ctx=num_ctx, seed=seed, timeout=timeout)

    def _generate(self, messages: list[Message], max_tokens: int = 1024, temperature: float = 0.0,
                  logprobs: bool = False, **kw) -> Generation:
        extra = {k: kw[k] for k in ("format", "options") if k in kw}
        return self._model.generate(messages, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs,
                                    **extra)

    def describe(self) -> dict:
        return {**self._model.describe(), "backend": "ollama_cloud"}
