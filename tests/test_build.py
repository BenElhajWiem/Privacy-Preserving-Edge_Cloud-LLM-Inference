import json

import pytest

from gapdeleg.build.align import align, all_occurrences
from gapdeleg.build.batch import cost, parse_json
from gapdeleg.build.io import id_order
from gapdeleg.build.personas import make_persona
from gapdeleg.build.privfin import check_rewrite, numbers, spans_for
from gapdeleg.build.pupa_labels import apply_labels, freeze_splits
from gapdeleg.data.loaders import EXTERNAL
from gapdeleg.data.schema import Example
from gapdeleg.privacy.types import Label, Span


def test_align_exact_fuzzy_and_none():
    q = "Rachel Zheng met the Univeristy of Westminster board."
    assert align(q, "rachel zheng") == (0, 12, "exact")
    s, e, how = align(q, "university of westminster")
    assert how == "fuzzy" and q[s:e] == "Univeristy of Westminster"
    assert align(q, "Johnny Bay") is None
    assert all_occurrences("Ann and ann, Annabel", "ann") == [(0, 3), (8, 11)]


def test_persona_is_deterministic():
    assert make_persona(3) == make_persona(3) and make_persona(3) != make_persona(4)


def test_privfin_checks_and_spans():
    p = make_persona(1)
    orig = "Kylar buys 16 glasses at $5 each, every second one at 60% of the price. How much?"
    good = (f"I'm {p.name} from {p.employer}; I pay from {p.bank}, account {p.account}. I buy 16 glasses at $5 "
            f"each, every second one at 60% of the price. How much do I pay, {p.first}?")
    assert check_rewrite(orig, good, p) == []
    assert "number 60 missing" in check_rewrite(orig, good.replace("60%", "half"), p)
    assert "missing account" in check_rewrite(orig, good.replace(p.account, "my account"), p)
    spans = spans_for(good, p)
    cats = [(s.text, s.category) for s in spans]
    assert (p.name, "PERSON") in cats and (p.account, "ACCOUNT_NUMBER") in cats and ("$5", "MONEY") in cats
    assert ("16", "AMOUNT") in cats and (p.first, "PERSON") in cats
    assert all(s.label == Label.LOCAL_ONLY and good[s.start:s.end] == s.text for s in spans)
    starts = sorted((s.start, s.end) for s in spans)
    assert all(a[1] <= b[0] for a, b in zip(starts, starts[1:]))
    assert numbers("$1,200.50, 3 and 4.") == ["1200.50", "3", "4"]


def test_pupa_apply_labels():
    ex = Example("pupa-tnb-0", "Rachel Zheng works at H&R Technology in five days.", "ref",
                 [Span("rachel zheng"), Span("h&r technology"), Span("unseen unit")])
    result = {"ok": True, "text": json.dumps({
        "units": [{"text": "rachel zheng", "label": "local-only", "category": "PERSON", "generalized": ""},
                  {"text": "h&r technology", "label": "abstractable", "category": "ORGANIZATION",
                   "generalized": "a tech company"}],
        "missed": [{"text": "in five days", "label": "abstractable", "category": "DATE_TIME", "generalized": "soon"}]})}
    out = apply_labels(ex, result)
    got = {(s.text, s.label.value, s.category, s.start) for s in out.private_spans}
    assert ("Rachel Zheng", "local-only", "PERSON", 0) in got
    assert ("H&R Technology", "abstractable", "ORGANIZATION", 22) in got
    assert ("in five days", "abstractable", "DATE_TIME", 37) in got
    assert ("unseen unit", "local-only", "OTHER", -1) in got
    assert out.meta["unlabeled_units"] == ["unseen unit"] and len(out.meta["unaligned"]) == 1
    assert out.meta["generalized"]["H&R Technology"] == "a tech company"
    failed = apply_labels(ex, {"ok": False})
    assert all(s.label == Label.LOCAL_ONLY for s in failed.private_spans)


def test_splits_are_fixed_and_disjoint():
    new = [Example(f"pupa-new-{i}", "q") for i in range(664)]
    a, b = freeze_splits(new), freeze_splits(new)
    assert a == b
    s = a["pupa_new"]
    assert (len(s["dev"]), len(s["optimization"]), len(s["lora_train"])) == (150, 150, 364)
    assert not set(s["dev"]) & set(s["optimization"]) and not set(s["dev"]) & set(s["lora_train"])


def test_id_order_sorts_by_the_trailing_number():
    ids = ["privfin-dev-gsm960", "pupa-tnb-12", "privfin-dev-gsm50", "privmed-dev-3"]
    assert sorted(ids, key=id_order) == ["privmed-dev-3", "pupa-tnb-12", "privfin-dev-gsm50", "privfin-dev-gsm960"]


def test_batch_helpers():
    assert parse_json({"ok": True, "text": '{"a": 1}'}) == {"a": 1}
    assert parse_json({"ok": True, "text": ""}) is None and parse_json(None) is None
    assert cost("claude-sonnet-5-5", {"input_tokens": 1_000_000, "output_tokens": 100_000}) == pytest.approx(1.5)


@pytest.mark.skipif(not (EXTERNAL / "PRISM").exists(), reason="external/ not cloned")
def test_privmed_prism_build():
    from gapdeleg.build.personas import FEMALE, MALE
    from gapdeleg.build.privmed import build_prism
    exs, stats = build_prism()
    assert stats["examples"] == 40
    for ex in exs:
        assert all(ex.query[s.start:s.end] == s.text for s in ex.private_spans)
        assert ex.meta["persona"]["name"] in ex.query and ex.meta["original"] != ex.query
        assert 0 < ex.meta["age"] <= 95
        if ex.meta["age"] >= 18:
            assert ex.meta["persona"]["first"] in (FEMALE if ex.meta["sex"] == "female" else MALE)
        else:
            assert "year-old daughter" in ex.query or "year-old son" in ex.query
    assert {s.label for s in exs[0].private_spans} == {Label.LOCAL_ONLY, Label.ABSTRACTABLE}


def test_persona_sex_and_backward_compatibility():
    from gapdeleg.build.personas import FEMALE, MALE
    assert make_persona(5, sex="female").first in FEMALE and make_persona(5, sex="male").first in MALE
    assert make_persona(5) == make_persona(5, sex=None)


def test_kappa():
    from gapdeleg.build.review_pupa import kappa
    assert kappa([("a", "a"), ("b", "b")]) == 1.0
    assert kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)
    assert kappa([("a", "a"), ("a", "b"), ("b", "b"), ("b", "a")]) == pytest.approx(0.0)


def test_privfin_augment_marks_other_names_and_number_words():
    from gapdeleg.build.privfin import augment_spans
    p = make_persona(2)
    q = f"I'm {p.name}. My friend Ginger has three bags and I have one; twice as many as Ginger?"
    ex = Example("x", q, "1", spans_for(q, p), "B",
                 {"persona": p.to_dict(), "original": "Ginger has three bags and Amy has one."})
    out = augment_spans(ex)
    got = {(s.text, s.source) for s in out.private_spans}
    assert ("Ginger", "gsm8k-name") in got and ("three", "quantity-word") in got
    assert not any(s.text in ("one", "twice") for s in out.private_spans)
    assert all(q[s.start:s.end] == s.text for s in out.private_spans)
