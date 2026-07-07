#!/usr/bin/env python3
"""memory_consolidate.py, quality-based consolidation of the holographic fact store.

DESIGN GUARANTEE: this NEVER prunes by age. A fact's created_at is not an input.
Older legitimate memories are never removed and their recall never degrades, in
fact recall improves, because removing duplicate/noise entries raises the signal
in both the vector index and the knowledge graph.

Two operations on memory_store.db facts ONLY (the curated MEMORY.md / USER.md layer
is never touched):

  1. Dedup (lossless): facts with identical content collapse to the single best
     copy (most helpful/retrieved/trusted, tie-broken to the OLDEST id).
  2. Junk archival: raw call-transcript auto-extractions that have NEVER proven
     useful. A fact is archived only if it matches an explicit transcript marker
     AND is not protected.

PROTECTED, never archived, regardless of age:
  - retrieval_count > 0   (the agent has recalled it)
  - helpful_count  > 0    (rated helpful)
  - trust_score  > default (boosted above the default, curated/reinforced)
Safety for the junk path rests on the deliberately NARROW transcript markers: a
fact is only "junk" if it is an explicit raw call-transcript dump, whose useful
content was already extracted into separate clean facts. The DURABLE pattern is
kept as a guard for any future, broader junk heuristics, it is intentionally
NOT applied to raw transcripts (which almost always contain an incidental number).

Removals are ARCHIVED to ~/.hermes/memory_archive.jsonl first (append-only,
reversible) and only then deleted, so nothing is ever truly lost (and the git
backup keeps a copy too). The 15m memory-ingest cron (memstore_sync) then prunes
from the vector index automatically.

Usage:
  memory_consolidate.py            # DRY RUN, report only, change nothing
  memory_consolidate.py --apply    # archive + remove the candidates
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
DB = HERMES_HOME / "memory_store.db"
ARCHIVE = HERMES_HOME / "memory_archive.jsonl"


def _default_trust() -> float:
    try:
        import yaml
        cfg = yaml.safe_load((HERMES_HOME / "config.yaml").read_text(encoding="utf-8-sig")) or {}
        return float(cfg.get("plugins", {}).get("hermes-memory-store", {}).get("default_trust", 0.5))
    except Exception:
        return 0.5


DEFAULT_TRUST = _default_trust()

# Explicit, unambiguous auto-extraction noise (raw voice-call dumps). Intentionally
# narrow: we only archive things that clearly are NOT intended memories.
JUNK_MARKERS = (
    "Voice call just ended",
    "Direction: inboundPhoneCall",
    "Direction: outboundPhoneCall",
    "Direction: outboundPhone",
)

# If a fact looks like real, durable information, it is protected outright.
DURABLE = re.compile(
    r"(\+?\d[\d\-\(\)\.\s]{7,}\d)"                 # phone number
    r"|([\w.+-]+@[\w-]+\.[\w.]+)"                  # email
    r"|(\b\d{1,2}/\d{1,2}(/\d{2,4})?\b)"           # date
    r"|(\bbirthday\b|\banniversary\b)"
    r"|(\d{1,3}%|\bequity\b|\bowns?\b|\bvesting\b|\bshares?\b)"
    r"|(\bprefer|\balways\b|\bnever\b|\bfavou?rite\b|\bdefault\b)",
    re.IGNORECASE,
)


def is_junk(text: str) -> bool:
    t = text or ""
    return any(m in t for m in JUNK_MARKERS)


def is_protected(row: sqlite3.Row) -> bool:
    """Usefulness-based protection. Applied to the junk path so that a raw
    transcript is kept only if the agent ever actually used it. DURABLE is NOT
    consulted here (raw call dumps incidentally contain numbers); it remains
    available for future broader junk heuristics."""
    if (row["retrieval_count"] or 0) > 0:
        return True
    if (row["helpful_count"] or 0) > 0:
        return True
    if (row["trust_score"] or 0) > DEFAULT_TRUST:
        return True
    return False


def norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def main():
    apply = "--apply" in sys.argv
    if not DB.exists():
        print("no memory_store.db, nothing to do")
        return

    con = sqlite3.connect(str(DB), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    rows = con.execute(
        "SELECT fact_id, content, category, tags, trust_score, "
        "retrieval_count, helpful_count, created_at FROM facts"
    ).fetchall()
    rowmap = {r["fact_id"]: r for r in rows}

    to_archive = {}  # fact_id -> reason

    # 1) Lossless dedup, identical content collapses to the best (oldest on ties).
    groups = {}
    for r in rows:
        k = norm(r["content"])
        if k:
            groups.setdefault(k, []).append(r)
    for k, g in groups.items():
        if len(g) < 2:
            continue
        keeper = sorted(g, key=lambda r: (
            -(r["helpful_count"] or 0),
            -(r["retrieval_count"] or 0),
            -(r["trust_score"] or 0),
            r["fact_id"],
        ))[0]
        for r in g:
            if r["fact_id"] != keeper["fact_id"]:
                to_archive[r["fact_id"]] = "duplicate"

    # 2) Junk archival, explicit transcript noise that is NOT protected.
    for r in rows:
        if r["fact_id"] in to_archive:
            continue
        if is_junk(r["content"]) and not is_protected(r):
            to_archive[r["fact_id"]] = "junk:call-transcript"

    kept = len(rows) - len(to_archive)
    by_reason = {}
    for reason in to_archive.values():
        by_reason[reason] = by_reason.get(reason, 0) + 1
    print(f"facts: {len(rows)} | keep: {kept} | archive: {len(to_archive)} "
          f"(default_trust={DEFAULT_TRUST})")
    for reason, c in sorted(by_reason.items()):
        print(f"  - {reason}: {c}")
    for fid, reason in list(to_archive.items())[:15]:
        print(f"    #{fid} [{reason}] {(rowmap[fid]['content'] or '')[:60]!r}")

    decision_report(con)

    if not apply:
        print("\nDRY RUN, nothing changed. Re-run with --apply to archive.")
        return
    if not to_archive:
        print("nothing to archive.")
        return

    stamp = datetime.now(timezone.utc).isoformat()
    removed = 0
    with open(ARCHIVE, "a", encoding="utf-8") as af:
        for fid, reason in to_archive.items():
            r = rowmap[fid]
            af.write(json.dumps({
                "archived_at": stamp, "reason": reason, "fact_id": fid,
                "content": r["content"], "category": r["category"], "tags": r["tags"],
                "trust_score": r["trust_score"], "created_at": r["created_at"],
            }, ensure_ascii=False) + "\n")
            try:
                con.execute("DELETE FROM facts WHERE fact_id=?", (fid,))  # facts_ad trigger cleans FTS
                con.execute("DELETE FROM fact_entities WHERE fact_id=?", (fid,))
                removed += 1
            except Exception as e:
                print(f"  [warn] could not remove #{fid}: {e}")
        con.commit()

    # Integrity check: FTS row count should track the facts table.
    nf = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    try:
        nfts = con.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
        fts_note = "ok" if nfts == nf else f"MISMATCH facts={nf} fts={nfts}"
    except Exception:
        fts_note = "n/a"
    print(f"\nArchived+removed {removed} facts -> {ARCHIVE}")
    print(f"facts now: {nf} | FTS integrity: {fts_note}")
    print("(memory-ingest cron prunes these from the vector index within 15m)")


def decision_report(con):
    """Weekly decision-trace summary (Phase 3, 2026-07-06): acceptance rate by
    kind plus recent rejections, the "decision context" panel at personal-agent
    scale. Read-only."""
    try:
        rows = con.execute(
            "SELECT kind, outcome, COUNT(*) FROM decision_log "
            "WHERE ts >= datetime('now', '-90 days') GROUP BY kind, outcome").fetchall()
    except sqlite3.OperationalError:
        return
    if not rows:
        return
    by_kind = {}
    for kind, outcome, n in rows:
        by_kind.setdefault(kind or "recommendation", {})[outcome] = n
    print("\nDecision traces (90d):")
    for kind, oc in sorted(by_kind.items()):
        resolved = sum(v for k, v in oc.items() if k in ("accepted", "rejected", "modified"))
        acc = oc.get("accepted", 0) + oc.get("modified", 0)
        rate = f"{100 * acc / resolved:.0f}%" if resolved else "n/a"
        detail = ", ".join(f"{k}={v}" for k, v in sorted(oc.items()))
        print(f"  {kind}: acceptance {rate} ({detail})")
    rej = con.execute(
        "SELECT ts, proposal, reason FROM decision_log WHERE outcome='rejected' "
        "AND ts >= datetime('now', '-90 days') ORDER BY id DESC LIMIT 5").fetchall()
    for ts, proposal, reason in rej:
        why = f" - {reason}" if reason else ""
        print(f"  rejected {str(ts)[:10]}: {(proposal or '')[:70]}{why}")


if __name__ == "__main__":
    main()
