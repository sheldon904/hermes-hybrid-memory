"""Entity-candidate hygiene added 2026-07-02 (hybrid plugin)."""

import importlib.util
from pathlib import Path

HYBRID = Path(__file__).resolve().parent.parent.parent / "plugins" / "hybrid" / "__init__.py"

spec = importlib.util.spec_from_file_location("hybrid_entity_filter_under_test", HYBRID)
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)

clean = hybrid._clean_entity_candidates


def test_newline_blobs_dropped():
    blob = "s travel schedule 2026:\n- July 17-19: Chicago\n- July 29"
    assert clean([blob, "John Doe"]) == ["John Doe"]


def test_lowercase_multiword_blobs_dropped():
    # apostrophe-rule spans surface lowercase with many words even w/o newlines
    assert clean(["s partner confirmed by Alex directly via voice call"]) == []


def test_overlong_dropped():
    assert clean(["A" * 61]) == []
    assert clean(["A" * 60]) == ["A" * 60]


def test_junk_lead_word_stripped_to_name():
    assert clean(["With Alex"]) == ["Alex"]
    assert clean(["Arrive Prague"]) == ["Prague"]


def test_junk_lead_word_without_name_dropped():
    assert clean(["With the"]) == []  # remainder not capitalized -> dropped


def test_good_candidates_kept_and_deduped():
    got = clean(["Acme Construction", "Czech Republic", "acme construction"])
    assert got == ["Acme Construction", "Czech Republic"]


def test_short_quoted_terms_survive():
    # single-word quoted tools like 'pytest' must not be killed by the
    # lowercase rule (it only fires on >3-word spans)
    assert clean(["pytest"]) == ["pytest"]
