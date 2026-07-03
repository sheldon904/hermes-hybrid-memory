#!/usr/bin/env python3
"""backfill_gists.py, extract gists for the whole existing fact store.

Loops memstore_sync.extract_gists_for_new_facts (batch 25, flash-lite) until
the gist_extracted watermark covers every fact. Resumable: interrupt any time
and re-run; only unwatermarked facts hit the LLM.

Usage:
  backfill_gists.py [--limit N]   # stop after N facts (default: all)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memstore as ms  # noqa: E402
import memstore_sync as msync  # noqa: E402


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    con = ms.connect()
    ms.init_schema(con)
    total = 0
    while True:
        n = msync.extract_gists_for_new_facts(con)
        if n <= 0:
            break
        total += n
        remaining = con.execute(
            "SELECT COUNT(*) FROM facts WHERE fact_id NOT IN "
            "(SELECT fact_id FROM gist_extracted)").fetchone()[0]
        print(f"  gisted {total} so far, {remaining} remaining")
        if limit and total >= limit:
            break
    ng = con.execute("SELECT COUNT(*) FROM fact_gists").fetchone()[0]
    nv = con.execute("SELECT COUNT(*) FROM vec_items WHERE type='gist'").fetchone()[0]
    print(f"done: {total} newly gisted | fact_gists={ng} | gist vectors={nv}")
    con.close()


if __name__ == "__main__":
    main()
