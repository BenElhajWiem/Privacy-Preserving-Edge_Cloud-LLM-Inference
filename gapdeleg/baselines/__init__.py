from __future__ import annotations

import re

from gapdeleg.confidence import MeanLogprob
from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace, strip_think, user
from gapdeleg.privacy.types import Label
from gapdeleg.split.template import mask

MASKABLE = (Label.LOCAL_ONLY, Label.ABSTRACTABLE)
MASK_INSTRUCTION = ("Placeholders like [PERSON_1] stand for hidden private values. Keep them exactly as "
                    "written in your answer.\n\n")


def masked_cloud_answer(ex: Example, ctx: Context, trace: Trace, labels=MASKABLE) -> str | None:
    spans = trace.label(ctx, ex.query)
    tpl = mask(ex.query, spans, labels=labels)
    trace.notes.update(n_spans=len(spans), masked_query=tpl.masked, placeholders=len(tpl.mapping))
    g = trace.cloud_call(ctx, "cloud_answer", user(MASK_INSTRUCTION + tpl.masked), spans)
    return None if g is None else tpl.restore(strip_think(g.text))


class EdgeOnly:
    name = "edge_only"
    needs_cloud = False

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        return strip_think(trace.edge_call(ctx, "edge_answer", user(ex.query), logprobs=True).text)


class CloudFull:
    name = "cloud_full"
    needs_cloud = True
    guard_disabled = True

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        g = trace.cloud_call(ctx, "cloud_answer", user(ex.query), [])
        return strip_think(g.text) if g else ""


class RedactSend:
    name = "redact_send"
    needs_cloud = True

    def __init__(self, labels=MASKABLE):
        self.labels = labels

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        out = masked_cloud_answer(ex, ctx, trace, self.labels)
        if out is None:
            trace.notes["fallback"] = "edge_only"
            return strip_think(trace.edge_call(ctx, "edge_fallback", user(ex.query), logprobs=True).text)
        return out


VERIFY_PROMPT = """Request:
{request}

Response:
{response}

Instruction: evaluate whether the response correctly and fully fulfils the request. Give a short reason,
then end with exactly one line: "Verdict: Correct" or "Verdict: Incorrect"."""


def verdict_correct(text: str) -> bool:
    text = strip_think(text).lower()
    found = re.findall(r"verdict\s*:\s*\**\s*(incorrect|correct)", text) or re.findall(r"\b(incorrect|correct)\b", text)
    return bool(found) and found[-1] == "correct"


class AutoMixGated:
    name = "automix"
    needs_cloud = True

    def __init__(self, k: int = 4, threshold: float = 0.5, temperature: float = 0.7, labels=MASKABLE):
        self.k, self.threshold, self.temperature, self.labels = k, threshold, temperature, labels

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        draft = strip_think(trace.edge_call(ctx, "edge_answer", user(ex.query), logprobs=True).text)
        prompt = user(VERIFY_PROMPT.format(request=ex.query, response=draft))
        votes = [verdict_correct(trace.edge_call(ctx, f"verify_{i}", prompt, logprobs=False, max_tokens=256,
                                                 temperature=self.temperature, seed=i + 1).text)
                 for i in range(self.k)]
        score = sum(votes) / len(votes) if votes else 0.0
        trace.notes.update(verification_score=score, verification_votes=votes)
        if score >= self.threshold:
            return draft
        out = masked_cloud_answer(ex, ctx, trace, self.labels)
        if out is None:
            trace.notes["fallback"] = "edge answer kept (guard denied)"
            return draft
        return out


class Cascade:
    name = "cascade"
    needs_cloud = True

    def __init__(self, threshold: float = 0.75, labels=MASKABLE):
        self.threshold, self.labels = threshold, labels

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        g = trace.edge_call(ctx, "edge_answer", user(ex.query), logprobs=True)
        draft = strip_think(g.text)
        try:
            score = MeanLogprob().score(ex.query, g)
        except ValueError:
            score = 0.0
        trace.notes["confidence"] = score
        if score >= self.threshold:
            return draft
        out = masked_cloud_answer(ex, ctx, trace, self.labels)
        if out is None:
            trace.notes["fallback"] = "edge answer kept (guard denied)"
            return draft
        return out


def _papillon(**kw):
    from gapdeleg.baselines.papillon import PapillonPort
    return PapillonPort(**kw)


def _prism(**kw):
    from gapdeleg.baselines.prism import PrismPort
    return PrismPort(**kw)


SYSTEMS = {c.name: c for c in (EdgeOnly, CloudFull, RedactSend, AutoMixGated, Cascade)}
SYSTEMS.update(papillon=_papillon, prism=_prism)


def make_system(name: str, **kw):
    if name == "gap_aware":
        from gapdeleg.pipeline.run import Flags, GapAwareDelegation
        return GapAwareDelegation(Flags(**kw))
    if name == "method_request":
        from gapdeleg.pipeline.method_request import MethodFlags, MethodRequest
        return MethodRequest(MethodFlags(**kw))
    if ":" in name:
        import importlib
        module, factory = name.split(":", 1)
        return getattr(importlib.import_module(module), factory)(**kw)
    return SYSTEMS[name](**kw)
