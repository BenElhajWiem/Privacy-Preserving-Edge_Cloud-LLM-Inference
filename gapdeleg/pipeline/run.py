from __future__ import annotations

from dataclasses import dataclass

from gapdeleg.baselines import MASK_INSTRUCTION, MASKABLE
from gapdeleg.confidence import MeanLogprob
from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace, strip_think, user
from gapdeleg.pipeline.gap_report import GapReport
from gapdeleg.privacy.types import Label, Span
from gapdeleg.split.template import Template, leftover_placeholders, mask


@dataclass
class Flags:
    split: bool = True
    confidence_gate: bool = True
    gap_loop: bool = True
    gaps: str = "report"
    max_rounds: int = 1
    max_gaps: int = 3
    trust_may_leave: bool = False
    threshold: float = 0.75
    rewrite: str = "full"


GAP_REPORT_PROMPT = """Here is a request and a draft answer. Private values are hidden behind placeholders like [PERSON_1].

Request:
{request}

Draft answer:
{draft}

Return JSON with three lists: "established" (points the draft already handles well), "uncertain"
(points that may be wrong), "missing" (what the draft lacks). Each item is {{"text": ..., "may_leave": true|false}};
set may_leave to false if asking about it would reveal a private detail. JSON only."""

CRITIQUE_PROMPT = """Request:
{request}

Draft answer:
{draft}

Briefly list what is wrong or missing in the draft."""

REWRITE_PROMPT = """Request:
{request}

Draft answer:
{draft}

Expert notes:
{notes}

Write the final answer to the request, using the notes where they help. Keep every placeholder like
[PERSON_1] exactly as written."""

REWRITE_FULL_PROMPT = """Request:
{request}

Draft answer:
{draft}

Expert notes:
{notes}

Write the final answer to the request. Use the notes where they improve the draft, and keep the names and
details of the request as they are. Answer with the final answer only."""


class GapAwareDelegation:
    name = "gap_aware"
    needs_cloud = True

    def __init__(self, flags: Flags | None = None):
        self.flags = flags or Flags()
        if self.flags.gaps not in ("report", "critique") or self.flags.rewrite not in ("full", "masked"):
            raise ValueError(f"unknown gaps={self.flags.gaps!r} or rewrite={self.flags.rewrite!r}")
        self.confidence = MeanLogprob()

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        f = self.flags
        spans = trace.label(ctx, ex.query)
        tpl = mask(ex.query, spans, labels=MASKABLE, cjk_words=True)
        trace.notes.update(masked_query=tpl.masked, n_spans=len(spans), placeholders=len(tpl.mapping))
        g = trace.edge_call(ctx, "edge_attempt", user(ex.query), logprobs=True)
        draft = strip_think(g.text)
        try:
            conf = self.confidence.score(ex.query, g)
        except ValueError:
            conf = 0.0
        trace.notes["confidence"] = conf
        if not f.confidence_gate or conf < f.threshold:
            c, tpl, spans = self._send(ex, ctx, trace, "cloud_r1", MASK_INSTRUCTION + tpl.masked, tpl, spans)
            answer = strip_think(c.text) if c else ""
            if answer:
                draft = tpl.restore(answer)
            else:
                trace.notes["fallback_r1"] = "guard denied" if c is None else "empty cloud answer"
        trace.notes["answer_before_gaps"] = draft
        for r in range(f.max_rounds if f.gap_loop else 0):
            gaps = [tpl.hide(x) for x in self._gaps(ctx, trace, r, tpl.masked, tpl.hide(draft))][:f.max_gaps]
            trace.notes[f"gaps_r{r}"] = gaps
            if not gaps:
                break
            ask = (f"{MASK_INSTRUCTION}Request:\n{tpl.masked}\n\nPlease help with these points:\n"
                   + "\n".join(f"- {x}" for x in gaps))
            c, tpl, spans = self._send(ex, ctx, trace, f"cloud_r{r + 2}", ask, tpl, spans)
            notes = strip_think(c.text) if c else ""
            if not notes:
                break
            draft = self._rewrite(ex, ctx, trace, r, tpl, draft, notes)
        return tpl.restore(draft)

    def _send(self, ex: Example, ctx: Context, trace: Trace, step: str, text: str, tpl: Template,
              spans: list[Span]):
        return guarded_send(ex, ctx, trace, step, text, tpl, spans)

    def _gaps(self, ctx: Context, trace: Trace, r: int, request: str, draft: str) -> list[str]:
        if self.flags.gaps == "report":
            s = trace.edge_call(ctx, f"gap_report_r{r}", user(GAP_REPORT_PROMPT.format(request=request, draft=draft)),
                                logprobs=False)
            report = GapReport.parse(s.text)
            trace.notes[f"gap_report_r{r}_parsed"] = report.parsed
            return report.gap_requests(only_may_leave=self.flags.trust_may_leave)
        s = trace.edge_call(ctx, f"critique_r{r}", user(CRITIQUE_PROMPT.format(request=request, draft=draft)),
                            logprobs=False)
        critique = strip_think(s.text)
        return [critique] if critique else []

    def _rewrite(self, ex: Example, ctx: Context, trace: Trace, r: int, tpl: Template, draft: str,
                 notes: str) -> str:
        if self.flags.rewrite == "full":
            prompt = REWRITE_FULL_PROMPT.format(request=ex.query, draft=draft, notes=tpl.restore(notes))
        else:
            prompt = REWRITE_PROMPT.format(request=tpl.masked, draft=tpl.hide(draft), notes=notes)
        out = tpl.restore(strip_think(trace.edge_call(ctx, f"rewrite_r{r}", user(prompt), logprobs=False).text))
        if not out or len(leftover_placeholders(out)) > len(leftover_placeholders(draft)):
            trace.notes[f"rewrite_r{r}_fallback"] = "draft kept"
            return draft
        return out


def guarded_send(ex: Example, ctx: Context, trace: Trace, step: str, text: str, tpl: Template, spans: list[Span],
                 request: str | None = None):
    request = ex.query if request is None else request
    c = trace.cloud_call(ctx, step, user(text), spans, request=request)
    found = ctx.cloud.last_decision.offending if c is None else []
    if found:
        tpl = tpl.extend(found)
        spans = spans + [Span(v, Label.LOCAL_ONLY, "PRIVATE", source="guard") for v in found]
        trace.notes.setdefault("remasked", []).append(step)
        c = trace.cloud_call(ctx, f"{step}_remasked", user(tpl.hide(text)), spans, request=request)
    return c, tpl, spans
