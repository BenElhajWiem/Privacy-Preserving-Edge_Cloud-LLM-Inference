#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gapdeleg.formula.formula_sandbox import compile_program
from gapdeleg.metrics.leakage import request_leakage
from gapdeleg.pipeline.method_request import format_number, formula_mask, to_number
from gapdeleg.privacy.guard import EgressGuard
from gapdeleg.privacy.policy import Policy
from gapdeleg.privacy.types import Label, Span
from gapdeleg.split.template import mask


def show(title):
    print(f"\n=== {title}")


def user(text):
    return [{"role": "user", "content": text}]


def family_a(guard):
    request = ("Hi, I am Maria Lopez and I work at Northwind Bank in Lyon. My account number is 4821-7730-1195. "
               "Please write a short email to my manager asking for Friday off.")
    spans = [Span("Maria Lopez", Label.LOCAL_ONLY, "PERSON"),
             Span("4821-7730-1195", Label.LOCAL_ONLY, "ACCOUNT_NUMBER"),
             Span("Northwind Bank", Label.ABSTRACTABLE, "ORGANIZATION"),
             Span("Lyon", Label.ABSTRACTABLE, "LOCATION"),
             Span("Friday", Label.PUBLIC, "DATE_TIME")]
    show("Family A: a request with private details")
    print(request)
    print("\nOn-device labels (given here; the method gets them from the edge model and the detectors):")
    for s in spans:
        print(f"  {s.label.value:12s} {s.category:15s} {s.text}")

    template = mask(request, spans)
    show("What may leave the device")
    print(template.masked)

    show("Egress guard")
    for name, text in (("full request", request), ("masked request", template.masked)):
        d = guard.check(user(text), spans)
        print(f"  {name:15s} -> {'allowed' if d.allow else 'blocked'}" + ("" if d.allow else f" ({d.reasons[0]})"))

    show("Leakage over everything that left the device")
    for name, text in (("full request", request), ("masked request", template.masked)):
        leak = request_leakage("demo-a", spans, [text], query=request)
        leaked = [s.text for s in leak.spans if s.leaked and s.label == Label.LOCAL_ONLY.value]
        print(f"  {name:15s} -> hard violation: {leak.hard_violation}  local-only values leaked: {leaked}")

    reply = (f"Subject: Day off on Friday\n\nDear manager,\nI would like to take Friday off. "
             f"Best regards,\n{next(p for p, v in template.mapping.items() if v == 'Maria Lopez')}")
    show("A cloud reply written over placeholders, restored on the device")
    print(template.restore(reply))


def family_b(guard):
    request = ("I'm Ines Asante. I earn $2400 a month at Kestrel Pharmacy and pay $650 in rent. "
               "How much do I keep over 3 months?")
    spans = [Span("Ines Asante", Label.LOCAL_ONLY, "PERSON"),
             Span("Kestrel Pharmacy", Label.LOCAL_ONLY, "ORGANIZATION")]
    show("Family B: the private numbers are the hard part")
    print(request)

    template, hidden = formula_mask(request, spans)
    show("Value-free method request sent to the cloud")
    print(template.masked)
    d = guard.check(user(template.masked), hidden)
    print(f"Egress guard: {'allowed' if d.allow else 'blocked'}")

    program = "def solve(NUM_1, NUM_2, NUM_3):\n    return (NUM_1 - NUM_2) * NUM_3"
    show("Program returned by the cloud (it never saw a value)")
    print(program)

    values = {p.strip("[]"): to_number(v) for p, v in template.mapping.items() if p.startswith("[NUM_")}
    answer = compile_program(program, template.mapping).run(values)
    show("Run in the on-device sandbox with the private values")
    print(f"values: {values}")
    print(f"Answer: {format_number(answer)}")


def main():
    guard = EgressGuard(Policy.load())
    family_a(guard)
    family_b(guard)


if __name__ == "__main__":
    main()
