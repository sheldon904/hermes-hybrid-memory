#!/usr/bin/env python3
"""Add source_ref provenance columns to facts and edges (Phase 1, 2026-07-06).

Idempotent: skips columns that already exist. Safe with the facts FTS5
triggers (they reference only content/tags).
"""
import os
import sqlite3
from pathlib import Path

DB = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "memory_store.db"


def main():
    con = sqlite3.connect(str(DB), timeout=30)
    try:
        for table in ("facts", "edges"):
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if "source_ref" in cols:
                print(f"{table}.source_ref already present, skipping")
                continue
            con.execute(f"ALTER TABLE {table} ADD COLUMN source_ref TEXT DEFAULT ''")
            print(f"added {table}.source_ref")
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    main()
