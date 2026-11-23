from __future__ import annotations

import time

from .base import Generation, Message, Usage


class MLXBackend:
    def __init__(self, model: str, adapter_path: str | None = None, enable_thinking: bool = False):
        from mlx_lm import load
        self.name = model if not adapter_path else f"{model}+{adapter_path}"
        self.model_id, self.adapter_path = model, adapter_path
        self.enable_thinking = enable_thinking
        self.model, self.tokenizer = load(model, adapter_path=adapter_path)

    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.0,
                 logprobs: bool = True, **kw) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
        mx.reset_peak_memory()
        text, lps, toks, last = [], [], [], None
        t = time.time()
        for last in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=temperature)):
            text.append(last.text)
            toks.append(last.text)
            if logprobs:
                lps.append(last.logprobs[last.token].item())
        wall = time.time() - t
        return Generation(
            text="".join(text),
            token_logprobs=lps if logprobs else None,
            tokens=toks,
            usage=Usage(last.prompt_tokens if last else 0, last.generation_tokens if last else 0),
            latency_s=wall,
            model=self.name,
            meta={"eval_tok_per_s": last.generation_tps if last else None,
                  "peak_memory_gb": mx.get_peak_memory() / 1e9},
        )

    def describe(self) -> dict:
        return {"backend": "mlx", "model": self.model_id, "adapter": self.adapter_path}
