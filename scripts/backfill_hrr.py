#!/usr/bin/env python3
"""backfill_hrr.py, recompute HRR vectors + category banks for every fact.

One-shot repair for facts written before the HRR-enabled pipeline came online
(2026-06-29): they have NULL hrr_vector and are therefore invisible to the
category banks and the analogy slot. Safe to re-run any time (idempotent, 
vectors are deterministic functions of content + entities).

Run inside the Hermes venv:
  HERMES_HOME=~/.hermes ~/.hermes/hermes-agent/venv/bin/python3 \
      ~/.hermes/scripts/backfill_hrr.py
"""

import os
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "hermes-agent"))


def main():
    from plugins.memory.holographic.store import MemoryStore

    db = HERMES_HOME / "memory_store.db"
    store = MemoryStore(db_path=str(db))
    before = store._conn.execute(
        "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NULL").fetchone()[0]
    print(f"facts with NULL hrr_vector before: {before}")
    n = store.rebuild_all_vectors()
    after = store._conn.execute(
        "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NULL").fetchone()[0]
    print(f"rebuilt {n} vectors; NULL hrr_vector after: {after}")
    print("banks:")
    for name, count in store._conn.execute(
            "SELECT bank_name, fact_count FROM memory_banks ORDER BY bank_name"):
        print(f"  {name}: {count}")
    store.close()


if __name__ == "__main__":
    main()
