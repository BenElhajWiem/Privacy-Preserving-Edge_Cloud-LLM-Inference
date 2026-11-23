from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

Message = dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Generation:
    text: str
    token_logprobs: list[float] | None = None
    tokens: list[str] | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""
    cached: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Generation":
        d = dict(d)
        d["usage"] = Usage(**d.get("usage", {}))
        return cls(**d)


@runtime_checkable
class EdgeModel(Protocol):
    name: str

    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.0,
                 logprobs: bool = True, **kw) -> Generation: ...

    def describe(self) -> dict: ...


def messages_digest(messages: list[Message]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class UnguardedEgressError(RuntimeError):
    pass


class CloudModel:

    name: str = "cloud"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.spent = Usage()

    def generate(self, messages: list[Message], *, ticket, max_tokens: int = 1024,
                 temperature: float = 0.0, logprobs: bool = False, **kw) -> Generation:
        from gapdeleg.privacy.guard import EgressTicket
        if not isinstance(ticket, EgressTicket) or not ticket.covers(messages):
            raise UnguardedEgressError("cloud call without a matching EgressTicket")
        params = {"max_tokens": max_tokens, "temperature": temperature, "logprobs": logprobs, **kw}
        key = self._cache_key(messages, params)
        if self.cache_dir and (hit := self._cache_get(key)):
            hit.cached = True
            return hit
        t = time.time()
        gen = self._generate(messages, **params)
        gen.latency_s = gen.latency_s or time.time() - t
        gen.model = gen.model or self.name
        self.spent.input_tokens += gen.usage.input_tokens
        self.spent.output_tokens += gen.usage.output_tokens
        if self.cache_dir:
            self._cache_put(key, gen)
        return gen

    def _generate(self, messages: list[Message], **params) -> Generation:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": type(self).__name__, "model": self.name}

    def _cache_key(self, messages, params) -> str:
        blob = json.dumps({"model": self.name, "messages": messages, "params": params},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> Generation | None:
        p = self._cache_path(key)
        return Generation.from_dict(json.loads(p.read_text())) if p.exists() else None

    def _cache_put(self, key: str, gen: Generation) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(gen.to_dict(), ensure_ascii=False))
        os.replace(tmp, p)


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    return system, [m for m in messages if m["role"] != "system"]
