"""Pure-function behavior of calendar_graph_sync (no network, no DB)."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "calendar_graph_sync_under_test", SCRIPTS / "calendar_graph_sync.py")
cgs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgs)


def test_event_node_name_embeds_start_date():
    ev = {"summary": "Trip to Czech Republic", "start": "2026-08-13T00:00:00-04:00"}
    assert cgs.event_node_name(ev) == "Trip to Czech Republic (2026-08-13)"


def test_event_node_name_all_day_and_missing_bits():
    assert cgs.event_node_name({"summary": "BIRTHDAY", "start": "2026-08-18"}) == "BIRTHDAY (2026-08-18)"
    assert cgs.event_node_name({"start": "2026-08-18"}) == "(no title) (2026-08-18)"
    assert cgs.event_node_name({"summary": "X"}) == "X"


def test_is_stale_only_for_future_personal_events():
    fetched = {"abc": "Party (2026-08-09)"}
    today = "2026-07-02"
    live = {"cal": "personal", "eventId": "abc", "start": "2026-08-09"}
    assert not cgs.is_stale(live, fetched, "Party (2026-08-09)", today)
    # renamed -> old node stale
    assert cgs.is_stale(live, fetched, "Party OLD (2026-08-09)", today)
    # vanished from window -> stale
    gone = {"cal": "personal", "eventId": "zzz", "start": "2026-08-09"}
    assert cgs.is_stale(gone, fetched, "Gone (2026-08-09)", today)
    # past events are history, never pruned
    past = {"cal": "personal", "eventId": "zzz", "start": "2026-06-01"}
    assert not cgs.is_stale(past, fetched, "Old (2026-06-01)", today)
    # non-calendar event nodes untouched (e.g. LLM-extracted episodes)
    llm = {"start": "2026-08-30"}
    assert not cgs.is_stale(llm, fetched, "Product demo", today)
