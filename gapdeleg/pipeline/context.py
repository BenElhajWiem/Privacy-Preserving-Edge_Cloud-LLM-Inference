from __future__ import annotations

import re
from dataclasses import dataclass, field

from gapdeleg.backends.base import Generation
from gapdeleg.pipeline.egress import GuardedCloud
from gapdeleg.privacy.labeler import Labeler
from gapdeleg.privacy.policy import Policy


@dataclass
class Context:
    edge: object
    cloud: GuardedCloud | None
    labeler: Labeler | None
    policy: Policy
    edge_max_tokens: int = 1024
    cloud_max_tokens: int = 1024


@dataclass
class Trace:
    request_id: str
    edge: list[tuple[str, Generation]] = field(default_factory=list)
    cloud: list[tuple[str, Generation]] = field(default_factory=list)
    guard_denied: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def label(self, ctx: Context, text: str) -> list:
        if ctx.labeler is None:
            return []
        spans = ctx.labeler.label(text)
        for g in getattr(ctx.labeler, "drain_generations", list)():
            self.edge.append(("labeler", g))
        return spans

    def edge_call(self, ctx: Context, step: str, messages, **kw) -> Generation:
        kw.setdefault("max_tokens", ctx.edge_max_tokens)
        g = ctx.edge.generate(messages, **kw)
        self.edge.append((step, g))
        return g

    def cloud_call(self, ctx: Context, step: str, messages, spans, request: str | None = None,
                   **kw) -> Generation | None:
        if ctx.cloud is None:
            raise RuntimeError("this system has no cloud model configured")
        kw.setdefault("max_tokens", ctx.cloud_max_tokens)
        if request is not None:
            kw["request"] = request
        g = ctx.cloud.send(messages, spans, request_id=self.request_id, step=step, **kw)
        check = getattr(getattr(ctx.cloud, "guard", None), "llm_check", None)
        for eg in getattr(check, "drain_generations", list)():
            self.edge.append((f"guard_{step}", eg))
        if g is None:
            self.guard_denied.append({"step": step, "reasons": ctx.cloud.last_decision.reasons})
        else:
            self.cloud.append((step, g))
        return g


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def user(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]
