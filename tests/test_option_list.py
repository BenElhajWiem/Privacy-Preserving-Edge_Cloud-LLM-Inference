from types import SimpleNamespace

from gapdeleg.metrics.option_list import leakage_without_options, masks_an_option, strip_options
from gapdeleg.privacy.types import Label, Span

OPTIONS = ["common cold", "malaria", "peptic ulcer disease"]
QUERY = "I feel tired and cold and I have a fever. Which of these: common cold, malaria, peptic ulcer disease?"


def span(text):
    start = QUERY.find(text)
    return Span(text=text, label=Label.LOCAL_ONLY, category="MEDICAL_CONDITION", start=start, end=start + len(text))


EX = SimpleNamespace(id="x", query=QUERY, meta={"options": OPTIONS}, private_spans=[span("cold"), span("cold"), span("fever")])


def test_strip_removes_whole_names_in_any_case():
    assert "cold" not in strip_options("Tell apart Common Cold, malaria.", OPTIONS).lower()
    assert "cold" in strip_options("I feel cold, not a common cold.", OPTIONS)
    assert strip_options("malarial", OPTIONS) == "malarial"


def test_the_option_list_alone_is_no_violation():
    assert not leakage_without_options(EX, ["Which is likely: common cold, malaria, peptic ulcer disease?"]).hard_violation


def test_a_symptom_sent_outside_the_list_still_counts():
    assert leakage_without_options(EX, ["I feel cold and I have a fever. Options: common cold, malaria."]).hard_violation


def test_no_options_means_the_metric_as_measured():
    ex = SimpleNamespace(**{**vars(EX), "meta": {}})
    assert leakage_without_options(ex, ["common cold and fever"]).hard_violation


def test_a_masked_word_in_the_list_is_found_and_the_plain_list_is_not():
    assert not masks_an_option("Tell apart: common cold, malaria, peptic ulcer disease.", OPTIONS)
    assert masks_an_option("Tell apart: common [PRIVATE_1], malaria, peptic ulcer disease.", OPTIONS)
    assert masks_an_option("Tell apart: common cold, [PRIVATE_2], peptic ulcer disease.", OPTIONS)
    assert not masks_an_option("A sum of [PRIVATE_1] dollars.", [])
