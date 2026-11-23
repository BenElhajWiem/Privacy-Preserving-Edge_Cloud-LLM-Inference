from __future__ import annotations

import os
import time

from .base import CloudModel, Generation, Message, Usage


DEFAULT_JUDGE = os.environ.get("GAPDELEG_JUDGE_MODEL", "gpt-5.5")


class OpenAIBackend(CloudModel):
    def __init__(self, model: str = DEFAULT_JUDGE, base_url: str | None = None, reasoning_effort: str = "low",
                 cache_dir=None):
        super().__init__(cache_dir)
        self.name = model
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self._client = None

    @property
    def client(self):
        if self._client is None:
            key = os.environ.get("GAPDELEG_OPENAI_API_KEY")
            if not key:
                raise RuntimeError("GAPDELEG_OPENAI_API_KEY is not set")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    @property
    def reasoning_model(self) -> bool:
        return self.name.startswith(("gpt-5", "o1", "o3", "o4"))

    def _generate(self, messages: list[Message], max_tokens: int = 1024, temperature: float = 0.0,
                  logprobs: bool = False, **kw) -> Generation:
        t = time.time()
        args = {"model": self.name, "messages": messages}
        if self.reasoning_model:
            args.update(max_completion_tokens=max_tokens, reasoning_effort=self.reasoning_effort)
        else:
            args.update(max_tokens=max_tokens, temperature=temperature, logprobs=logprobs or None)
        r = self.client.chat.completions.create(**args)
        ch = r.choices[0]
        lp = ch.logprobs.content if logprobs and ch.logprobs else None
        return Generation(
            text=ch.message.content or "",
            token_logprobs=[x.logprob for x in lp] if lp else None,
            tokens=[x.token for x in lp] if lp else None,
            usage=Usage(r.usage.prompt_tokens, r.usage.completion_tokens),
            latency_s=time.time() - t,
            model=r.model,
        )

    def describe(self) -> dict:
        d = {"backend": "openai", "model": self.name, "base_url": self.base_url}
        if self.reasoning_model:
            d["reasoning_effort"] = self.reasoning_effort
        return d
