import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pupa_components as pc

from gapdeleg.privacy.types import Label, Span
from gapdeleg.split.template import mask


def test_auroc_perfect_reversed_ties_and_one_class():
    assert pc.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert pc.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert pc.auroc([0.5, 0.5], [1, 0]) == 0.5
    assert pc.auroc([0.3, 0.4], [1, 1]) is None


def test_ece_calibrated_and_overconfident():
    assert pc.ece([0.25] * 4, [1, 0, 0, 0]) == pytest.approx(0.0)
    assert pc.ece([0.95] * 4, [0, 0, 0, 0]) == pytest.approx(0.95)
    assert pc.ece([], []) is None


@pytest.mark.parametrize("query,spans", [
    ("Write to Mark Smith at Acme about the $60 fee.", ["Mark Smith", "Acme", "$60"]),
    ("Mark and mark again, then Acme.", ["Mark", "Acme"]),
    ("Acme Corp", ["Acme Corp"]),
    ("Dear Ana Lopez, your id AB-1234 is ready", ["Ana", "Lopez", "AB-1234"]),
])
def test_masked_stretches_recover_what_mask_hid(query, spans):
    tpl = mask(query, [Span(s, Label.LOCAL_ONLY, "PERSON") for s in spans], cjk_words=True)
    got = pc.masked_stretches(query, tpl.masked)
    assert got is not None
    hidden = "".join(query[a:b] for a, b in got)
    for s in spans:
        assert s.replace(" ", "") in hidden.replace(" ", "") or s.lower() in hidden.lower()
    kept, pos = [], 0
    for a, b in got:
        kept.append(query[pos:a])
        pos = b
    kept.append(query[pos:])
    assert "".join(kept) == pc.PLACEHOLDER.sub("", tpl.masked)


def test_masked_stretches_unmasked_and_misaligned():
    assert pc.masked_stretches("nothing private", "nothing private") == []
    assert pc.masked_stretches("nothing private", "something else") is None
    assert pc.masked_stretches("Hi Bob", "Bye [PERSON_1]") is None


def test_occurrences_and_overlaps():
    q = "Mark met mark; Marked is another word."
    occ = pc.occurrences(q, "Mark")
    assert occ == [(0, 4), (9, 13)]
    assert pc.overlaps((2, 3), occ) and not pc.overlaps((15, 21), occ)


def test_check_of_reason_prefixes():
    assert pc.check_of("local-only span (PERSON) found [exact]") == "known value"
    assert pc.check_of("detector: EMAIL_ADDRESS (presidio, 1.00)") == "detector"
    assert pc.check_of("request check: ID from the request found") == "request check"
    assert pc.check_of("local model: private value found") == "leak check"
    assert pc.check_of("guard error: ValueError: x") == "error"
