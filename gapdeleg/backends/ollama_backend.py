from __future__ import annotations

import json
import os
import time
import urllib.request

from .base import Generation, Message, Usage

DEFAULT_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")


class OllamaBackend:
    def __init__(self, model: str, host: str = DEFAULT_HOST, think: bool = False,
                 num_ctx: int = 4096, timeout: float = 600, seed: int | None = 0):
        self.name = model
        self.host = host.rstrip("/")
        self.think = think
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.seed = seed
        self._info: dict | None = None

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(f"{self.host}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.host}{path}", timeout=30) as r:
            return json.load(r)

    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.0,
                 logprobs: bool = True, top_logprobs: int = 0, **kw) -> Generation:
        options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = kw.pop("seed", self.seed)
        options.update(kw.pop("options", {}))
        body = {"model": self.name, "messages": messages, "stream": False, "options": options,
                "think": self.think}
        if kw.get("format"):
            body["format"] = kw.pop("format")
        if logprobs:
            body["logprobs"] = True
            if top_logprobs:
                body["top_logprobs"] = top_logprobs
        t = time.time()
        r = self._post("/api/chat", body)
        wall = time.time() - t
        lps = r.get("logprobs") or None
        gen_s = r.get("eval_duration", 0) / 1e9
        return Generation(
            text=r["message"]["content"],
            token_logprobs=[x["logprob"] for x in lps] if lps else None,
            tokens=[x["token"] for x in lps] if lps else None,
            usage=Usage(r.get("prompt_eval_count", 0), r.get("eval_count", 0)),
            latency_s=wall,
            model=self.name,
            meta={"eval_tok_per_s": r.get("eval_count", 0) / gen_s if gen_s else None,
                  "load_s": r.get("load_duration", 0) / 1e9,
                  "top_logprobs": [x.get("top_logprobs") for x in lps] if lps and top_logprobs else None},
        )

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float | None = None,
                 stop: list[str] | None = None) -> Generation:
        options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        if top_p is not None:
            options["top_p"] = top_p
        if stop:
            options["stop"] = stop
        t = time.time()
        r = self._post("/api/generate", {"model": self.name, "prompt": prompt, "raw": True, "stream": False,
                                         "options": options})
        gen_s = r.get("eval_duration", 0) / 1e9
        return Generation(text=r.get("response", ""), usage=Usage(r.get("prompt_eval_count", 0), r.get("eval_count", 0)),
                          latency_s=time.time() - t, model=self.name,
                          meta={"eval_tok_per_s": r.get("eval_count", 0) / gen_s if gen_s else None, "raw": True})

    def describe(self) -> dict:
        if self._info is None:
            tags = self._get("/api/tags").get("models", [])
            m = next((t for t in tags if t["name"] == self.name or t["model"] == self.name), None)
            if m is None:
                raise RuntimeError(f"Ollama model {self.name} is not pulled (run: ollama pull {self.name})")
            version = self._get("/api/version").get("version")
            self._info = {"backend": "ollama", "model": self.name, "digest": m["digest"],
                          "size_bytes": m.get("size"), "details": m.get("details", {}),
                          "ollama_version": version, "think": self.think,
                          "host": self.host}
        return self._info
