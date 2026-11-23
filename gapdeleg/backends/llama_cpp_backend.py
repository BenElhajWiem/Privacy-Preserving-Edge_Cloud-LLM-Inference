from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from .base import Generation, Message, Usage


class LlamaCppBackend:
    def __init__(self, model_path: str, n_ctx: int = 4096):
        from llama_cpp import Llama
        self.path = Path(model_path).expanduser()
        self.name = self.path.name
        self.llm = Llama(model_path=str(self.path), n_ctx=n_ctx, n_threads=os.cpu_count(),
                         n_gpu_layers=-1, logits_all=True, verbose=False)

    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.0,
                 logprobs: bool = True, **kw) -> Generation:
        t = time.time()
        r = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                            temperature=temperature, logprobs=logprobs,
                                            top_logprobs=1 if logprobs else None)
        ch = r["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") if logprobs else None
        return Generation(
            text=ch["message"]["content"],
            token_logprobs=[x["logprob"] for x in lp] if lp else None,
            tokens=[x["token"] for x in lp] if lp else None,
            usage=Usage(r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]),
            latency_s=time.time() - t,
            model=self.name,
        )

    def describe(self) -> dict:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            h.update(f.read(1 << 24))
        return {"backend": "llama_cpp", "model": self.name, "path": str(self.path),
                "sha256_first16mb": h.hexdigest()}
