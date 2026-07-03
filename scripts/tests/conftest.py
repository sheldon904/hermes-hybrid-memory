"""Shared fixtures for the memory-stack tests.

Run with the Hermes venv:
  ~/.hermes/hermes-agent/venv/bin/python3 -m pytest ~/.hermes/scripts/tests/ -q

Doubles as the post-upstream-update smoke suite: if hermes-agent updates and
the holographic base class shifts, these tests catch the breakage before the
gateway does.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
HERMES_AGENT = SCRIPTS.parent / "hermes-agent"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERMES_AGENT))


@pytest.fixture()
def mem_db(tmp_path):
    """A throwaway unified-store DB with the full memstore schema."""
    import memstore as ms
    con = ms.connect(tmp_path / "memory_store.db")
    ms.init_schema(con)
    yield con
    con.close()


@pytest.fixture()
def facts_db(tmp_path):
    """A throwaway DB with a minimal holographic `facts` + entity schema and
    the memstore tables, for scripts that read both."""
    import sqlite3
    import memstore as ms
    path = tmp_path / "memory_store.db"
    con = ms.connect(path)
    ms.init_schema(con)
    con.executescript("""
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            aliases TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE fact_entities (
            fact_id INTEGER, entity_id INTEGER,
            PRIMARY KEY (fact_id, entity_id)
        );
    """)
    con.commit()
    yield path, con
    con.close()


def add_fact(con, content, category="general", created_days_ago=10, entities=()):
    cur = con.execute(
        "INSERT INTO facts(content, category, created_at) "
        "VALUES (?,?, datetime('now', ?))",
        (content, category, f"-{created_days_ago} days"))
    fid = cur.lastrowid
    for ent in entities:
        row = con.execute("SELECT entity_id FROM entities WHERE name=?", (ent,)).fetchone()
        eid = row[0] if row else con.execute(
            "INSERT INTO entities(name) VALUES (?)", (ent,)).lastrowid
        con.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?,?)",
                    (fid, eid))
    con.commit()
    return fid
