from __future__ import annotations

import os
import time

from .base import CloudModel, Generation, Message, Usage, split_system

DEFAULT_MODEL = os.environ.get("GAPDELEG_CLOUD_MODEL", "claude-sonnet-5-5")


class AnthropicBackend(CloudModel):
    def __init__(self, model: str = DEFAULT_MODEL, effort: str | None = "low", cache_dir=None):
        super().__init__(cache_dir)
        self.name = model
        self.effort = effort
        self._client = None

    @property
    def client(self):
        if self._client is None:
            key = os.environ.get("GAPDELEG_ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("GAPDELEG_ANTHROPIC_API_KEY is not set (see .env.example)")
            import anthropic
            self._client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")
        return self._client

    def _generate(self, messages: list[Message], max_tokens: int = 1024, temperature: float = 0.0,
                  logprobs: bool = False, **kw) -> Generation:
        system, msgs = split_system(messages)
        args = {"model": self.name, "max_tokens": max_tokens, "messages": msgs}
        if system:
            args["system"] = system
        if self.effort:
            args["output_config"] = {"effort": self.effort}
        t = time.time()
        r = self.client.messages.create(**args)
        return Generation(
            text="".join(b.text for b in r.content if b.type == "text"),
            usage=Usage(r.usage.input_tokens, r.usage.output_tokens),
            latency_s=time.time() - t,
            model=r.model,
            meta={"stop_reason": r.stop_reason},
        )

    def describe(self) -> dict:
        return {"backend": "anthropic", "model": self.name, "effort": self.effort}
