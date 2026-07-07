#!/usr/bin/env python3
"""One-time relation-vocabulary remap + idea->job retype (Phase 3, 2026-07-06).

Collapses the relation sprawl into the closed vocabulary via
entity_resolve.canon_relation (single source of truth shared with the live
writer, so migration and ingest can never disagree). Every changed edge keeps
its original relation string in rel_orig, making any cluster mechanically
reversible. System-written rows (proposed/rejected/alias-candidate/chunk/
calendar tags, POSSIBLE_ALIAS/INSTANCE_OF rels) are untouched.

Idempotent: a second run finds nothing to change.

Usage:
  2026-07-relation-remap.py --dry-run   # print the mapping table, write nothing
  2026-07-relation-remap.py             # apply
"""
import os
import re
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "scripts"))

import memstore as ms  # noqa: E402
from entity_resolve import canon_relation  # noqa: E402

SKIP_TAGS = ("proposed", "rejected", "alias-candidate", "chunk", "calendar")
SKIP_RELS = ("POSSIBLE_ALIAS", "INSTANCE_OF")

JOB_TITLE_RE = re.compile(
    r"(?i)\b(engineer|developer|architect|manager|lead|consultant|analyst|"
    r"designer|scientist|technician|director|intern|recruiter)\b")
JOB_EDGE_RELS = ("APPLIED_TO", "HIRING", "POSTED_ON")


def remap_relations(con, apply=True):
    """Returns (mapping_counts, moved, dropped_dupes, dropped_self)."""
    qm_tags = ",".join("?" * len(SKIP_TAGS))
    qm_rels = ",".join("?" * len(SKIP_RELS))
    rows = con.execute(
        f"SELECT src, rel, dst, src_tag, source_ref, rel_orig FROM edges "
        f"WHERE (src_tag IS NULL OR src_tag NOT IN ({qm_tags})) "
        f"AND rel NOT IN ({qm_rels})",
        (*SKIP_TAGS, *SKIP_RELS)).fetchall()
    mapping = {}
    moved = dupes = selfs = 0
    for src, rel, dst, tag, ref, orig in rows:
        canon, flipped = canon_relation(rel)
        if canon == rel and not flipped:
            continue
        mapping.setdefault((rel, canon, flipped), [0])[0] += 1
        if not apply:
            continue
        new_src, new_dst = (dst, src) if flipped else (src, dst)
        if new_src == new_dst:
            con.execute("DELETE FROM edges WHERE src=? AND rel=? AND dst=?",
                        (src, rel, dst))
            selfs += 1
            continue
        cur = con.execute(
            "INSERT OR IGNORE INTO edges(src, rel, dst, src_tag, source_ref, rel_orig) "
            "VALUES (?,?,?,?,?,?)",
            (new_src, canon, new_dst, tag, ref or "", (orig or rel)[:60]))
        if cur.rowcount:
            moved += 1
        else:
            dupes += 1  # canonical row already existed; its provenance wins
        con.execute("DELETE FROM edges WHERE src=? AND rel=? AND dst=?",
                    (src, rel, dst))
    if apply:
        con.commit()
    return mapping, moved, dupes, selfs


def retype_nodes(con, apply=True):
    """idea -> job for nodes in the jobs pipeline or matching a title pattern."""
    qm = ",".join("?" * len(JOB_EDGE_RELS))
    to_retype = [r[0] for r in con.execute(
        f"SELECT name FROM graph_nodes WHERE type='idea' AND ("
        f"  name IN (SELECT dst FROM edges WHERE rel IN ({qm}))"
        f"  OR name IN (SELECT src FROM edges WHERE rel IN ({qm}))"
        f")", (*JOB_EDGE_RELS, *JOB_EDGE_RELS))]
    by_title = [r[0] for r in con.execute(
        "SELECT name FROM graph_nodes WHERE type='idea'")
        if JOB_TITLE_RE.search(r[0] or "") and r[0] not in set(to_retype)]
    to_retype += by_title
    if apply and to_retype:
        con.executemany("UPDATE graph_nodes SET type='job' WHERE name=?",
                        [(n,) for n in to_retype])
        con.commit()
    return to_retype


def main():
    dry = "--dry-run" in sys.argv
    con = ms.connect()
    ms.init_schema(con)  # guarantees rel_orig column

    def _stats():
        return (con.execute("SELECT COUNT(DISTINCT rel) FROM edges").fetchone()[0],
                con.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    d0, e0 = _stats()
    print(f"BEFORE: {d0} distinct relations, {e0} edges")

    mapping, moved, dupes, selfs = remap_relations(con, apply=not dry)
    print(f"\n{'DRY-RUN ' if dry else ''}mapping ({len(mapping)} distinct source relations):")
    for (old, new, flip), (n,) in sorted(mapping.items(), key=lambda x: -x[1][0]):
        arrow = "-> (flipped)" if flip else "->"
        print(f"  {old} {arrow} {new}  ({n})")

    retyped = retype_nodes(con, apply=not dry)
    print(f"\n{'DRY-RUN ' if dry else ''}idea->job retype: {len(retyped)} nodes")
    for n in retyped[:20]:
        print(f"  {n}")
    if len(retyped) > 20:
        print(f"  ... and {len(retyped) - 20} more")

    if dry:
        print("\nDRY RUN, nothing written.")
        con.close()
        return 0

    d1, e1 = _stats()
    print(f"\nAFTER: {d1} distinct relations, {e1} edges "
          f"(moved {moved}, merged-into-existing {dupes}, self-edges dropped {selfs})")
    if d1 > 40:
        leftovers = [r[0] for r in con.execute(
            "SELECT DISTINCT rel FROM edges").fetchall()]
        print(f"WARN: {d1} distinct relations still > 40: {sorted(leftovers)}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
