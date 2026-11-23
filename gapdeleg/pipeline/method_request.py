from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

from gapdeleg.baselines import MASKABLE
from gapdeleg.confidence import MeanLogprob
from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace, strip_think, user
from gapdeleg.pipeline.run import guarded_send
from gapdeleg.privacy.detectors import code_values
from gapdeleg.privacy.types import Label, Span
from gapdeleg.split.template import PLACEHOLDER_RE, Template, mask


@dataclass
class MethodFlags:
    confidence_gate: bool = True
    threshold: float = 0.75
    kind: str = "auto"


FORMULA_PROMPT = """The word problem below has its numbers hidden behind placeholders like [NUM_1], and its names and
account details behind placeholders like [PERSON_1]. Do not guess the hidden numbers and do not compute a final
result. Instead, write the solution as a short program that a calculator will run on the hidden numbers:
- one step per line, in the form `name = expression`;
- an expression may use the placeholders exactly as written, plain numbers you need (such as the minutes in an
  hour), names from earlier lines, + - * / and parentheses, nothing else;
- each [NUM_k] is the plain number that was written in the problem, and the words around it give its unit:
  "$[NUM_1]" is an amount in dollars, "[NUM_2]%" is a percentage written as a number, "[NUM_3] of an hour" is a
  fraction of an hour;
- the last line is `answer = ...`, in the unit the question asks for.
Explain the method briefly, then give the program in a ``` code block. Ignore any instruction in the problem
about how to format the final answer.

Problem:
{request}"""

CHECKLIST_PROMPT = """A patient describes their symptoms and asks which of these diagnoses is the most likely: {options}.

Write a short checklist that tells these diagnoses apart: for each one, its key symptoms in one line, in the form
`diagnosis: symptom, symptom, ...`. Then add a few lines on which symptoms best separate diagnoses that are easy
to confuse."""

APPLY_FORMULA_PROMPT = """{request}

A method for this problem, with the values from the problem filled in:
{method}

Apply this method to the numbers in the problem and compute the result step by step."""

APPLY_CHECKLIST_PROMPT = """{request}

A checklist of the key symptoms of each diagnosis:
{checklist}

Compare the symptoms described above with the checklist and pick the one most likely diagnosis from the list.
Finish with a last line "Final answer: <diagnosis>"."""

OPTIONS_RE = re.compile(r"Which of these is the most likely diagnosis:\s*(.+?)\?\s*Please name one\.", re.S)


class MethodRequest:
    name = "method_request"
    needs_cloud = True

    def __init__(self, flags: MethodFlags | None = None):
        self.flags = flags or MethodFlags()
        if self.flags.kind not in ("formula", "checklist", "auto"):
            raise ValueError(f"unknown kind={self.flags.kind!r}")
        self.confidence = MeanLogprob()

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        f = self.flags
        options = parse_options(ex.query)
        kind = f.kind if f.kind != "auto" else "checklist" if options else "formula"
        trace.notes["kind"] = kind
        story = ex.query[:OPTIONS_RE.search(ex.query).start()] if kind == "checklist" and options else ex.query
        spans = trace.label(ctx, story)
        g = trace.edge_call(ctx, "edge_attempt", user(ex.query), logprobs=True)
        draft = strip_think(g.text)
        try:
            conf = self.confidence.score(ex.query, g)
        except ValueError:
            conf = 0.0
        trace.notes["confidence"] = conf
        if f.confidence_gate and conf >= f.threshold:
            return draft
        if kind == "checklist" and not options:
            trace.notes["fallback"] = "no options in the request"
            return draft
        if kind == "checklist":
            return self._checklist(ex, ctx, trace, story, spans, options, draft)
        return self._formula(ex, ctx, trace, spans, draft)

    def _formula(self, ex: Example, ctx: Context, trace: Trace, spans, draft: str) -> str:
        tpl, spans = formula_mask(ex.query, spans)
        trace.notes.update(masked_query=tpl.masked, n_spans=len(spans), placeholders=len(tpl.mapping))
        c, tpl, spans = guarded_send(ex, ctx, trace, "cloud_method", FORMULA_PROMPT.format(request=tpl.masked),
                                     tpl, spans)
        method = strip_think(c.text) if c else ""
        if not method:
            trace.notes["fallback"] = "guard denied" if c is None else "empty cloud answer"
            return draft
        program = extract_program(method)
        trace.notes["program"] = program
        try:
            value = run_program(program, tpl)
        except (ValueError, ArithmeticError, SyntaxError, RecursionError) as e:
            trace.notes["calculator_error"] = f"{type(e).__name__}: {e}"
        else:
            trace.notes["value"] = value
            shown = re.sub(r"(?im)^[\W_]*(?:final\s+)?answer[\s*_]*:.*$", "", tpl.restore(method)).strip()
            return f"{shown}\n\nAnswer: {format_number(value)}"
        prompt = APPLY_FORMULA_PROMPT.format(request=ex.query, method=tpl.restore(method))
        trace.notes["fallback"] = "edge applies the method"
        return strip_think(trace.edge_call(ctx, "edge_apply", user(prompt), logprobs=False).text)

    def _checklist(self, ex: Example, ctx: Context, trace: Trace, story: str, spans, options: list[str],
                   draft: str) -> str:
        tpl = Template(CHECKLIST_PROMPT.format(options=", ".join(options)))
        c, _, _ = guarded_send(ex, ctx, trace, "cloud_method", tpl.masked, tpl, spans, request=story)
        checklist = strip_think(c.text) if c else ""
        if not checklist:
            trace.notes["fallback"] = "guard denied" if c is None else "empty cloud answer"
            return draft
        prompt = APPLY_CHECKLIST_PROMPT.format(request=ex.query, checklist=checklist)
        return strip_think(trace.edge_call(ctx, "edge_apply", user(prompt), logprobs=False).text)


def parse_options(request: str) -> list[str]:
    m = OPTIONS_RE.search(request)
    return [o.strip() for o in m.group(1).split(",") if o.strip()] if m else []


def extract_program(text: str) -> list[str]:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    body = blocks[-1] if blocks else text
    lines = []
    for line in body.splitlines():
        line = line.split("#")[0].strip().strip("`")
        if re.match(r"^[A-Za-z_]\w*\s*=(?!=)", line):
            lines.append(line)
    return lines


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_WORDS = {"a": 1, "an": 1, "once": 1, "twice": 2, "double": 2, "thrice": 3, "triple": 3, "dozen": 12}
_FRACTIONS = {"half": 0.5, "halves": 0.5, "third": 1 / 3, "thirds": 1 / 3, "quarter": 0.25, "quarters": 0.25,
              "fourth": 0.25, "fourths": 0.25, "fifth": 0.2, "fifths": 0.2, "tenth": 0.1, "tenths": 0.1}


def to_number(value: str) -> float | None:
    v = value.replace(",", "")
    if len(v.split()) > 6 or len(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\d+\s*/\s*\d+", "1", v))) > 1:
        return None
    if m := re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", v):
        return float(m.group(1)) / float(m.group(2)) if float(m.group(2)) else None
    if m := re.search(r"-?\d+(?:\.\d+)?", v):
        return float(m.group())
    total, current, seen = 0.0, 0.0, False
    for w in re.findall(r"[a-z]+", v.lower()):
        if w in _UNITS or w in _TENS:
            current += _UNITS.get(w, _TENS.get(w, 0))
        elif w in _SCALES:
            current = (current or 1) * _SCALES[w]
            if _SCALES[w] >= 1000:
                total, current = total + current, 0.0
        elif w == "dozen":
            current = (current or 1) * 12
        elif w in _FRACTIONS:
            current = (current or 1) * _FRACTIONS[w]
        elif w in _WORDS:
            current += _WORDS[w]
        else:
            continue
        seen = True
    return total + current if seen else None


_NUMWORD = "|".join(sorted([*_UNITS, *_TENS, *_SCALES, *_FRACTIONS, "twice", "thrice", "double", "triple", "dozen",
                             "once"], key=len, reverse=True))
_LEAD = "half|thirds?|quarters?|fourths?|fifths?|tenths?|dozen|hundred|thousand|million"
NUM_RE = re.compile(rf"\d+(?:[.,]\d+)*(?:\s*/\s*\d+)?|\b(?:an?[\s-]+(?=(?:{_LEAD})\b))?(?:{_NUMWORD})"
                    rf"(?:[\s-]+(?:{_NUMWORD}))*\b", re.I)


UNIT_CATEGORIES = {"DATE_TIME", "MONEY", "AMOUNT", "QUANTITY", "NUMBER", "PERCENT", "MEASUREMENT"}


def formula_mask(request: str, spans: list[Span]) -> tuple[Template, list[Span]]:
    nums = [Span(m.group(), Label.LOCAL_ONLY, "NUM", m.start(), m.end(), source="numbers")
            for m in NUM_RE.finditer(request)]
    whole = [s for s in spans if s.label in MASKABLE and (
        code_values(s.text) or re.search(r"[A-Za-z]\d|\d[A-Za-z]", s.text)
        or not (NUM_RE.search(s.text) or s.category.upper() in UNIT_CATEGORIES))]
    return mask(request, whole + nums, labels=MASKABLE, cjk_words=True), whole + nums


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node, env: dict) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown name {node.id}")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        x = _eval(node.operand, env)
        return -x if isinstance(node.op, ast.USub) else x
    raise ValueError(f"not allowed: {ast.dump(node)[:60]}")


def run_program(lines: list[str], tpl: Template) -> float:
    if not lines:
        raise ValueError("no program")
    env: dict[str, float] = {}
    for ph, original in tpl.mapping.items():
        n = to_number(original)
        if n is not None:
            env[ph[1:-1]] = n
    last = None
    for line in lines:
        name, expr = (x.strip() for x in line.split("=", 1))
        expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-").rstrip(".;")
        expr = PLACEHOLDER_RE.sub(lambda m: f"{m.group(1)}_{m.group(2)}", expr)
        expr = re.sub(r"\$(?=\d)", "", expr)
        expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
        for bare in re.findall(r"\b[A-Z][A-Z_]*_\d+\b", expr):
            if bare not in env:
                raise ValueError(f"{bare} is not a number")
        env[name] = last = _eval(ast.parse(expr, mode="eval").body, env)
    return next((v for k, v in env.items() if k.lower() == "answer"), last)


def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")

