from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from gapdeleg.baselines import MASKABLE
from gapdeleg.confidence import MeanLogprob
from gapdeleg.data.schema import Example
from gapdeleg.metrics.quality import extract_number
from gapdeleg.pipeline.context import Context, Trace, strip_think, user
from gapdeleg.pipeline.method_request import NUM_RE, extract_program, format_number, formula_mask, to_number
from gapdeleg.pipeline.run import guarded_send
from gapdeleg.privacy.detectors import code_values
from gapdeleg.privacy.types import Label, Span
from gapdeleg.formula.formula_sandbox import NotANumber, RunError, StaticError, compile_program
from gapdeleg.split.template import Template, mask


@dataclass
class FormulaFlags:
    prompt: str = "py"
    mask: str = "v0"
    repairs: int = 1
    fallback: str = "apply"
    gate: float | None = None


PY_PROMPT = """The word problem below has its numbers hidden behind placeholders like [NUM_1], and its names and
account details behind placeholders like [PERSON_1]. Do not guess the hidden numbers and do not compute a final
result. Instead, write a Python function that solves the problem for whatever the hidden numbers are; a calculator
on the user's device will run it on the real numbers.
- Write `def solve(NUM_1, NUM_2, ...):` with a parameter for each hidden number the solution needs, named like
  its placeholder without the brackets ([NUM_3] becomes NUM_3). It returns the number the question asks for, in
  the unit the question asks for.
{types}
- Names, employers and account numbers are not numbers: never use their placeholders.
- Use only + - * / // % **, comparisons, if/else, for loops over range(), while loops, and `round`, `min`, `max`,
  `abs`, `int`, `math.floor`, `math.ceil`, `math.sqrt`. No other imports or functions, no strings, no print.
- If the question needs a whole number (whole items, people, trips or days), use math.floor or math.ceil as the
  situation requires; otherwise do not round.
Write a sentence on the method, then the function in a single ```python code block. Ignore any instruction in the
problem about how to format the final answer.
{example}
Problem:
{request}"""

TYPES = {
    "v0": """- Each parameter is the plain number written in the problem, in digits or as a word, and the words around
  its placeholder give its unit: "$[NUM_1]" is an amount in dollars; "[NUM_2]%" is the percent number as written
  before the % sign, not a fraction; "[NUM_3] of an hour" is a fraction of an hour. A hidden number can also be a
  word for a multiplier or a fraction; its value is then that factor, so "[NUM_4] as many" means multiply by
  NUM_4.""",
    "clean": """- [NUM_k] hides a number written in the problem, in digits or as a word, and the words around it give its
  unit: "$[NUM_1]" is an amount in dollars, and "[NUM_2]%" is the percent number as written before the % sign, not
  a fraction.
- [RATIO_k] hides a multiplier or a fraction (a word for so many times, or a part of something), as the factor
  itself. So "[RATIO_1] as many" means multiply by RATIO_1, "[RATIO_2] of them" means RATIO_2 times them, and
  "[RATIO_3] of an hour" is RATIO_3 hours.""",
}

EXAMPLE = """
Example, for a different problem:
Problem: Hi, I'm [PERSON_1] and I work at [ORGANIZATION_1]; I pay from my account [ACCOUNT_NUMBER_1]. Notebooks
cost $[NUM_1] each. I have $[NUM_2]. I first buy [NUM_3] notebooks for my team, and my manager wants [RATIO_1]
as many for the office. How many more notebooks can I buy for myself with the money left?
Reply: Pay for the team's and the office's notebooks, then see how many whole notebooks the rest of the money buys.
```python
import math

def solve(NUM_1, NUM_2, NUM_3, RATIO_1):
    bought = NUM_3 + NUM_3 * RATIO_1     # for the team and for the office
    left = NUM_2 - bought * NUM_1        # money left
    return math.floor(left / NUM_1)      # only whole notebooks
```
"""

REPAIR = """Your function could not run: {error}. Write the corrected function in a single ```python code block,
following the rules above."""

APPLY_PROMPT = """{request}

A method for this problem, written as a Python program by a helper who could not see the numbers. In it,
{bindings}.
```python
{code}
```
Apply this method to the numbers in the problem and compute the result step by step."""


def build_prompt(masked: str, prompt: str = "py", mask_mode: str = "v0") -> str:
    example = ""
    if prompt == "py_ex":
        example = EXAMPLE if mask_mode == "clean" else EXAMPLE.replace("RATIO_1", "NUM_4")
    return PY_PROMPT.format(types=TYPES[mask_mode], example=example, request=masked)


RATIO_RE = re.compile(r"\b(?:twice|thrice|double|triple|half|halves|thirds?|quarters?|fourths?|fifths?|tenths?)\b"
                      r"|\d\s*/\s*\d", re.I)
_STOP = set("""I I'm I've I'd I'll Hi Hello Hey Dear My Our Their His Her Its The A An This That These Those It We
He She They You Your Me Us Them How What When Where Which Who Whom Whose Why If And But So Or Nor Yet Then Also Now
Last Next Each Every Both All Any Some No Not After Before During While Since Until On In At For By To From With
Without Of As About Is Are Was Were Be Been Do Does Did Can Could Would Should Must Shall Might Please Thanks
Thank Today Yesterday Tomorrow Tonight Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February
March July September October November December Christmas Easter Halloween Thanksgiving Answer Yes Okay OK Here
There Let Let's""".split())
_CONNECT = {"of", "and", "&", "de", "du", "da", "van", "von", "der", "la", "le", "del"}
_WORD = re.compile(r"[A-Za-z0-9&][A-Za-z0-9&'’.-]*")
_NAME_TYPES = {"PERSON": "PERSON", "ORGANIZATION": "ORGANIZATION", "LOCATION": "LOCATION", "ADDRESS": "LOCATION"}


def _possessive(w: str) -> str:
    return re.sub(r"['’]s?$", "", w)


def name_runs(text: str) -> list[str]:
    runs, cur = [], []

    def flush():
        while cur and cur[-1][0] in _CONNECT:
            cur.pop()
        if cur:
            start, end = cur[0][1], cur[-1][2]
            name = _possessive(text[start:end])
            if len(name) > 1:
                runs.append(name)
        cur.clear()

    for m in _WORD.finditer(text):
        w = m.group().rstrip(".-'’") or m.group()
        end = m.start() + len(w)
        gap = text[cur[-1][2]:m.start()] if cur else ""
        if cur and gap.strip():
            flush()
        cap = w[:1].isupper() and w not in _STOP and _possessive(w) not in _STOP
        if cap or (w in _CONNECT and cur):
            cur.append((w, m.start(), end))
        else:
            flush()
    flush()
    return list(dict.fromkeys(runs))


def number_type(text: str) -> str:
    return "RATIO" if RATIO_RE.search(text) else "NUM"


def clean_mask(request: str, spans: list[Span]) -> tuple[Template, list[Span]]:
    nums = [Span(m.group(), Label.LOCAL_ONLY, number_type(m.group()), m.start(), m.end(), source="numbers")
            for m in NUM_RE.finditer(request)]
    whole, names = [], []
    for s in spans:
        if s.label not in MASKABLE:
            continue
        if code_values(s.text) or re.search(r"[A-Za-z]\d|\d[A-Za-z]", s.text):
            whole.append(s)
            continue
        typ = _NAME_TYPES.get(s.category.upper(), "NAME")
        names += [Span(n, Label.LOCAL_ONLY, typ, source=f"names-of-{s.source or 'labeler'}") for n in name_runs(s.text)]
    hidden = whole + names + nums
    return mask(request, hidden, labels=MASKABLE, cjk_words=True), hidden


def extract_code(reply: str) -> tuple[str, str]:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", reply, re.S)
    if blocks:
        with_def = [b for b in blocks if re.search(r"^\s*def\s", b, re.M)]
        return (with_def or blocks)[-1], "block"
    if m := re.search(r"```[^\n]*\n(.*)$", reply, re.S):
        return m.group(1), "unclosed block"
    if m := re.search(r"^def\s+\w+\s*\(.*", reply, re.M | re.S):
        return m.group(0), "bare def"
    return "\n".join(extract_program(reply)), "lines"


def method_line(reply: str) -> str:
    prose = re.sub(r"```.*?(```|$)", "", reply, flags=re.S)
    prose = re.sub(r"(?im)^[\W_]*(?:final\s+)?answer[\s*_]*:.*$", "", prose)
    return "\n".join(x for x in prose.splitlines() if x.strip()).strip()


def bind_values(tpl: Template) -> dict:
    out = {}
    for ph, original in tpl.mapping.items():
        name = ph[1:-1]
        n = to_number(original)
        out[name] = n if n is not None else NotANumber(name)
    return out


def send_messages(ex: Example, ctx: Context, trace: Trace, step: str, messages: list[dict], tpl: Template,
                  spans: list[Span]):
    c = trace.cloud_call(ctx, step, messages, spans, request=ex.query)
    found = ctx.cloud.last_decision.offending if c is None else []
    if found:
        tpl = tpl.extend(found)
        spans = spans + [Span(v, Label.LOCAL_ONLY, "PRIVATE", source="guard") for v in found]
        trace.notes.setdefault("remasked", []).append(step)
        messages = [m if m["role"] == "assistant" else {**m, "content": tpl.hide(m["content"])} for m in messages]
        c = trace.cloud_call(ctx, f"{step}_remasked", messages, spans, request=ex.query)
    return c, tpl, spans


class FormulaProgram:
    name = "formula_program"
    needs_cloud = True

    def __init__(self, flags: FormulaFlags | None = None):
        self.flags = flags or FormulaFlags()
        f = self.flags
        if f.prompt not in ("py", "py_ex") or f.mask not in ("v0", "clean") or f.fallback not in ("apply", "draft"):
            raise ValueError(f"unknown prompt, mask or fallback: {f}")
        self.confidence = MeanLogprob()

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        f, notes = self.flags, trace.notes
        notes.update(kind="formula-py", prompt=f.prompt, mask=f.mask)
        spans = trace.label(ctx, ex.query)
        g = trace.edge_call(ctx, "edge_attempt", user(ex.query), logprobs=True)
        draft = strip_think(g.text)
        try:
            conf = self.confidence.score(ex.query, g)
        except ValueError:
            conf = 0.0
        notes.update(confidence=conf, draft_number=extract_number(draft))
        if f.gate is not None and conf >= f.gate:
            notes["route"] = "draft (confident)"
            return draft

        tpl, gspans = formula_mask(ex.query, spans) if f.mask == "v0" else clean_mask(ex.query, spans)
        notes.update(masked_query=tpl.masked,
                     placeholder_types=dict(Counter(ph[1:-1].rsplit("_", 1)[0] for ph in tpl.mapping)),
                     hidden_words={ph: v for ph, v in tpl.mapping.items()
                                   if not ph.startswith(("[NUM_", "[RATIO_"))})
        c, tpl, gspans = guarded_send(ex, ctx, trace, "cloud_method", build_prompt(tpl.masked, f.prompt, f.mask),
                                      tpl, gspans)
        reply = strip_think(c.text) if c else ""
        if not reply:
            return self._draft(notes, draft, "guard denied" if c is None else "empty cloud reply")

        attempts, code = [], ""
        for k in range(f.repairs + 1):
            code, where = extract_code(reply)
            att = {"program": code, "where": where}
            attempts.append(att)
            notes["attempts"] = attempts
            try:
                prog = compile_program(code, tpl.mapping)
            except StaticError as e:
                att["static_error"] = str(e)
                if k == f.repairs:
                    break
                messages = [*user(build_prompt(tpl.masked, f.prompt, f.mask)), {"role": "assistant", "content": reply},
                            *user(REPAIR.format(error=re.sub(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", "#", str(e))))]
                c, tpl, gspans = send_messages(ex, ctx, trace, "cloud_repair", messages, tpl, gspans)
                reply = strip_think(c.text) if c else ""
                if not reply:
                    att["repair"] = "guard denied" if c is None else "empty cloud reply"
                    break
                continue
            att.update(mode=prog.mode, used=prog.used)
            try:
                value = prog.run(bind_values(tpl))
            except RunError as e:
                att["run_error"] = str(e)
                break
            att["value"] = value
            bindings = {u: tpl.mapping.get(f"[{u}]", "") for u in prog.used}
            notes.update(program=code, ran=True, value=value, repaired=k > 0, bindings=bindings, route="program")
            shown = tpl.restore(method_line(reply))
            values = ", ".join(f"{u} = {v}" for u, v in bindings.items())
            return (f"{shown}\n\n```python\n{code.strip()}\n```\nwith the numbers of the problem: {values}\n\n"
                    f"Answer: {format_number(value)}")

        last = attempts[-1]
        error = last.get("static_error") or last.get("run_error") or last.get("repair") or "no program"
        notes.update(program=code, ran=False, error=error, repaired=len(attempts) > 1, route="fallback")
        if f.fallback == "draft" or not code.strip():
            return self._draft(notes, draft, f"program did not run: {error}")
        used = sorted(set(re.findall(r"\b[A-Z][A-Z_]*_\d+\b", code)) & {ph[1:-1] for ph in tpl.mapping})
        bindings = "; ".join(f"{u} is {tpl.mapping[f'[{u}]']!r}" for u in used) or "no numbers are named"
        notes["bindings"] = {u: tpl.mapping[f"[{u}]"] for u in used}
        notes["fallback"] = f"edge applies the method (program did not run: {error})"
        prompt = APPLY_PROMPT.format(request=ex.query, bindings=bindings, code=code.strip())
        return strip_think(trace.edge_call(ctx, "edge_apply", user(prompt), logprobs=False).text)

    @staticmethod
    def _draft(notes: dict, draft: str, why: str) -> str:
        notes.setdefault("route", "fallback")
        notes["fallback"] = f"edge draft ({why})"
        return draft


def make(**kw) -> FormulaProgram:
    return FormulaProgram(FormulaFlags(**kw))
