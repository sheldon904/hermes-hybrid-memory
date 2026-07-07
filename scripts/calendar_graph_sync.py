#!/usr/bin/env python3
"""calendar_graph_sync.py, mirror upcoming calendar events into the knowledge graph.

Each event on the personal Google Calendar (the single source of truth)
becomes a first-class graph EPISODE: a node of type 'event' carrying
start/end/location/eventId attrs, wired with edges:

    <owner>  ATTENDS     <event>
    <event>  INVOLVES    <known person/company node named in the event text>
    <event>  LOCATED_IN  <known location node matched in the event text>

This is the calendar<->memory reconciliation: trips, parties, and flights are
queryable relationally (graph_query('Prague'), analogize) with real dates,
instead of living only as prose facts.

Idempotent: node name embeds the start date; attrs.eventId keys refresh and
rename/cancel handling. All edges carry src_tag='calendar' (deletable by tag).
Runs from the 15-min memory-ingest cron; no embedder is loaded (cheap).
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "scripts"))
import memstore as ms  # noqa: E402

GAPI = HERMES_HOME / "skills" / "productivity" / "google-workspace" / "scripts" / "google_api.py"
PY = HERMES_HOME / "hermes-agent" / "venv" / "bin" / "python3"
OWNER = os.environ.get("CALENDAR_OWNER_NODE", "Me")  # the graph's root identity node
WINDOW_PAST_DAYS = 1
WINDOW_FUTURE_DAYS = 60
CAL_TAG = "calendar"
# Generic titles that would create meaningless INVOLVES matches.
_MATCH_MIN_LEN = 4


def event_node_name(ev: dict) -> str:
    """Stable node name: summary + start date (keeps recurring events distinct)."""
    start = (ev.get("start") or "")[:10]
    summary = " ".join((ev.get("summary") or "(no title)").split())[:80]
    return f"{summary} ({start})" if start else summary


def is_stale(attrs: dict, fetched: dict, node_name: str, today: str) -> bool:
    """A calendar node is stale when its event vanished from the window or was
    renamed (same eventId now maps to a different node name). Only FUTURE
    events are pruned, past episodes stay as history."""
    if (attrs or {}).get("cal") != "personal":
        return False
    start = str(attrs.get("start") or "")[:10]
    if not start or start < today:
        return False
    eid = attrs.get("eventId")
    if not eid:
        return False
    return eid not in fetched or fetched[eid] != node_name


def fetch_events() -> list:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=WINDOW_PAST_DAYS)).isoformat()
    end = (now + timedelta(days=WINDOW_FUTURE_DAYS)).isoformat()
    out = subprocess.run(
        [str(PY), str(GAPI), "calendar", "list",
         "--start", start, "--end", end, "--max", "100"],
        capture_output=True, text=True, timeout=90)
    if out.returncode != 0:
        raise RuntimeError(f"calendar list failed: {(out.stderr or out.stdout)[:300]}")
    return json.loads(out.stdout)


def known_nodes(con):
    """Graph nodes usable for INVOLVES/LOCATED_IN matching, by lowercase name."""
    people, places = {}, {}
    for name, ntype in con.execute("SELECT name, type FROM graph_nodes"):
        if not name or len(name) < _MATCH_MIN_LEN:
            continue
        low = str(name).lower()
        if ntype in ("person", "company"):
            people[low] = name
        elif ntype == "location":
            places[low] = name
    return people, places


def main() -> int:
    con = ms.connect()
    ms.init_schema(con)
    try:
        events = fetch_events()
    except Exception as e:
        print(f"calendar_graph_sync: fetch failed: {e}", file=sys.stderr)
        return 1

    people, places = known_nodes(con)
    today = datetime.now(timezone.utc).date().isoformat()
    fetched = {}
    upserts = edges = 0

    for ev in events:
        if (ev.get("status") or "") == "cancelled":
            continue
        name = event_node_name(ev)
        eid = ev.get("id") or ""
        if eid:
            fetched[eid] = name
        attrs = {"cal": "personal", "eventId": eid,
                 "start": ev.get("start") or "", "end": ev.get("end") or ""}
        if ev.get("location"):
            attrs["location"] = ev["location"][:120]
        ms.add_node(con, name, "event", attrs)
        source_ref = f"gcal:{eid}"[:200] if eid else ""
        ms.add_edge(con, OWNER, "ATTENDS", name, CAL_TAG, source_ref=source_ref)
        upserts += 1
        edges += 1
        haystack = " ".join([ev.get("summary") or "", ev.get("description") or "",
                             ev.get("location") or ""]).lower()
        for low, canon in people.items():
            if low in haystack:
                ms.add_edge(con, name, "INVOLVES", canon, CAL_TAG, source_ref=source_ref)
                edges += 1
        for low, canon in places.items():
            if low in haystack:
                ms.add_edge(con, name, "LOCATED_IN", canon, CAL_TAG, source_ref=source_ref)
                edges += 1

    # Prune future events that were cancelled or renamed.
    removed = 0
    for node_name, attrs_json in con.execute(
            "SELECT name, attrs FROM graph_nodes WHERE type='event'").fetchall():
        try:
            attrs = json.loads(attrs_json) if attrs_json else {}
        except Exception:
            continue
        if is_stale(attrs, fetched, node_name, today):
            con.execute("DELETE FROM edges WHERE src=? OR dst=?", (node_name, node_name))
            con.execute("DELETE FROM graph_nodes WHERE name=?", (node_name,))
            removed += 1

    con.commit()
    con.close()
    print(f"calendar_graph_sync: {upserts} events upserted, {edges} edges, "
          f"{removed} stale removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
