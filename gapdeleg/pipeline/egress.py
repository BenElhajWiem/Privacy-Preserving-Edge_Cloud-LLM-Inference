from __future__ import annotations

import json
import time
from pathlib import Path

from gapdeleg.backends.base import CloudModel, Generation
from gapdeleg.privacy.guard import EgressGuard, GuardDecision
from gapdeleg.privacy.types import Span


class GuardedCloud:
    def __init__(self, cloud: CloudModel, guard: EgressGuard, outgoing_path: str | Path | None = None):
        self.cloud, self.guard = cloud, guard
        self.outgoing_path = Path(outgoing_path) if outgoing_path else None
        self.last_decision: GuardDecision | None = None

    def send(self, messages: list[dict], spans: list[Span], request_id: str, step: str = "",
             request: str | None = None, **gen_kw) -> Generation | None:
        decision = self.guard.check(messages, spans, request_id=request_id, request=request)
        self.last_decision = decision
        if not decision.allow:
            return None
        gen = self.cloud.generate(messages, ticket=decision.ticket, **gen_kw)
        self._log(request_id, step, messages, gen)
        return gen

    def _log(self, request_id, step, messages, gen: Generation):
        if not self.outgoing_path:
            return
        self.outgoing_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"t": time.time(), "request_id": request_id, "step": step, "model": gen.model,
               "messages": messages, "response": gen.text, "usage": gen.to_dict()["usage"],
               "latency_s": gen.latency_s, "cached": gen.cached}
        with open(self.outgoing_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_outgoing(path: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["request_id"], []).extend(m["content"] for m in r["messages"] if m["role"] != "assistant")
    return out
