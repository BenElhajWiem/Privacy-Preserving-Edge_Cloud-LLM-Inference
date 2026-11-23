from gapdeleg.privacy.types import Label, Span
from gapdeleg.split.template import leftover_placeholders, mask, restore


def S(text, cat="PERSON", label=Label.LOCAL_ONLY, **kw):
    return Span(text, label, cat, **kw)


def test_mask_and_restore_roundtrip():
    q = "Rachel Zheng works at H&R Technology. Rachel will email Johnny Bay."
    spans = [S("Rachel Zheng"), S("rachel"), S("H&R Technology", "ORGANIZATION", Label.ABSTRACTABLE),
             S("johnny bay")]
    t = mask(q, spans)
    assert "Rachel" not in t.masked and "Johnny" not in t.masked and "H&R" not in t.masked
    assert t.masked == "[PERSON_1] works at [ORGANIZATION_1]. [PERSON_2] will email [PERSON_3]."
    assert t.restore(t.masked) == q


def test_longest_span_wins_and_same_text_same_placeholder():
    q = "Johnny Bay called. Later Johnny Bay called again."
    t = mask(q, [S("Johnny"), S("Johnny Bay")])
    assert t.masked == "[PERSON_1] called. Later [PERSON_1] called again."
    assert t.mapping == {"[PERSON_1]": "Johnny Bay"}


def test_whole_word_only():
    t = mask("Al also likes Alabama.", [S("Al")])
    assert t.masked == "[PERSON_1] also likes Alabama."


def test_labels_filter():
    q = "Bob lives in Paris."
    t = mask(q, [S("Bob"), S("Paris", "LOCATION", Label.ABSTRACTABLE)], labels=(Label.LOCAL_ONLY,))
    assert t.masked == "[PERSON_1] lives in Paris."


def test_public_spans_never_masked():
    t = mask("The CEO of a bank", [S("CEO", "OCCUPATION", Label.PUBLIC)])
    assert t.masked == "The CEO of a bank"


def test_generic_category_placeholder():
    t = mask("Meet at Westminster", [S("westminster", "GENERIC")])
    assert t.masked == "Meet at [PRIVATE_1]"


def test_restore_tolerates_missing_brackets():
    assert restore("Dear PERSON_1, thanks", {"[PERSON_1]": "Rachel"}) == "Dear Rachel, thanks"
    assert restore("[PERSON_1] and PERSON_12", {"[PERSON_1]": "A"}) == "A and PERSON_12"


def test_offsets_are_used_when_given():
    q = "Ana met Ana."
    t = mask(q, [S("Ana", start=0, end=3)])
    assert t.masked == "[PERSON_1] met [PERSON_1]."


def test_leftover_placeholders():
    assert leftover_placeholders("hi [PERSON_1], from [ORGANIZATION_2]") == [("PERSON", "1"), ("ORGANIZATION", "2")]


def test_restore_keeps_backslashes():
    path = r"C:\Users\Abu Bakr\Desktop\Test.txt"
    assert restore("Open [PATH_1] and PATH_1.", {"[PATH_1]": path}) == f"Open {path} and {path}."


def test_hide_puts_placeholders_into_other_text():
    t = mask("Rachel Zheng works at H&R Technology.", [S("Rachel Zheng"), S("H&R Technology", "ORGANIZATION")])
    assert t.hide("Dear rachel zheng, H&R Technology is hiring. Rachelle") == \
        "Dear [PERSON_1], [ORGANIZATION_1] is hiring. Rachelle"
    assert mask("no spans", []).hide("Rachel") == "Rachel"


def test_extend_continues_the_numbering_and_restores():
    t = mask("Rachel Zheng emails Erin and Andy.", [S("Rachel Zheng")])
    t2 = t.extend(["Erin", "Andy"])
    assert t2.masked == "[PERSON_1] emails [PRIVATE_1] and [PRIVATE_2]."
    assert t2.mapping["[PERSON_1]"] == "Rachel Zheng" and t2.mapping["[PRIVATE_2]"] == "Andy"
    t3 = t2.extend(["Erin", "Rachel Zheng"])
    assert t3.mapping == t2.mapping and t3.masked == t2.masked
    assert t3.restore("Hi [PRIVATE_1], from [PERSON_1]") == "Hi Erin, from Rachel Zheng"
    assert t2.hide("ask Andy") == "ask [PRIVATE_2]"
