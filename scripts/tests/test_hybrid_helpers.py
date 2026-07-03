"""Pure-function behavior of the hybrid plugin (no gateway, no DB)."""

import importlib.util
import sys
from pathlib import Path

HYBRID = Path(__file__).resolve().parent.parent.parent / "plugins" / "hybrid" / "__init__.py"

spec = importlib.util.spec_from_file_location("hybrid_under_test", HYBRID)
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)


def test_cap_blocks_keeps_whole_lines():
    blocks = ["## A\n- one\n- two", "## B\n- three"]
    text, included = hybrid._cap_blocks(blocks, cap=0)  # cap 0 = disabled
    assert "- three" in text and len(included) == 3

    text, included = hybrid._cap_blocks(blocks, cap=16)  # only "## A\n- one" fits
    assert text == "## A\n- one"
    assert included == {"- one"}


def test_cap_blocks_priority_order():
    blocks = ["## High\n" + "\n".join(f"- h{i}" for i in range(10)),
              "## Low\n- l0"]
    text, included = hybrid._cap_blocks(blocks, cap=60)
    assert "- l0" not in included  # lower-priority block starved first
    assert any(l.startswith("- h") for l in included)


def test_cap_blocks_header_only_block_dropped():
    text, included = hybrid._cap_blocks(["## Empty"], cap=0)
    assert text == "" and not included


def test_content_words_strips_stopwords():
    out = hybrid._content_words("should I keep pouring the time into it")
    assert "the" not in out.split() and "should" not in out.split()
    assert "pouring" in out and "time" in out


def test_fact_id_from_vid():
    f = hybrid.HybridMemoryProvider._fact_id_from_vid
    assert f("holo_fact_42") == 42
    assert f("gist_7") == 7
    assert f("memory_abc123") is None
    assert f("") is None


def test_counter_cosine():
    from collections import Counter
    cos = hybrid.HybridMemoryProvider._counter_cosine
    a = Counter({("out", "CLIENT_OF", "company"): 1})
    assert cos(a, a) == 1.0
    assert cos(a, Counter({("in", "OWES", "person"): 1})) == 0.0
    assert cos(a, Counter()) == 0.0


def test_default_config_keys():
    cfg = hybrid._load_hybrid_config()
    for key in ("prefetch_char_cap", "operational_categories", "analogy_slot",
                "analogy_z_min", "situation_sim_min", "rolling_window"):
        assert key in cfg
