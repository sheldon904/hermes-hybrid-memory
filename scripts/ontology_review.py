#!/usr/bin/env python3
"""ontology_review.py, operator gate for machine-proposed graph changes.

Two proposal queues accumulate in memory_store.db:
  - POSSIBLE_ALIAS edges (src_tag='alias-candidate'), entity_resolve fuzzy
    near-misses drained by memory_ingest; "are these the same entity?"
  - INSTANCE_OF edges (src_tag='proposed') + category nodes, weekly
    memory_abstract fingerprint clustering; "is this a real category?"

Decisions are durable via the src_tag convention (see memstore.py):
  approve alias    -> Resolver.add_alias (future writes) + merge_nodes (graph)
  reject  alias    -> POSSIBLE_ALIAS edge flipped to src_tag='rejected'
  approve category -> INSTANCE_OF edges flipped to src_tag='curated'
  reject  category -> flipped to src_tag='rejected' (node kept, filtered)
Rejected edges are kept ON PURPOSE: their (src, rel, dst) primary key makes
any re-proposal an INSERT OR IGNORE no-op. load_graph() hides proposed/
rejected/alias-candidate tags from all readers, so an unreviewed proposal
never leaks into prefetch or a graph tool as if it were ground truth.

Library-first: functions take a memstore connection so the hybrid plugin's
`ontology_review` tool, the CLI, and tests share one implementation.

CLI:
  ontology_review.py list
  ontology_review.py approve alias "Loser Name" "Winner Name"
  ontology_review.py reject  alias "A" "B"
  ontology_review.py approve category "label"
  ontology_review.py reject  category "label"
  ontology_review.py --nudge     # weekly cron: summary if pending, else silent
"""
import os
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "scripts"))

import memstore as ms  # noqa: E402


def pending_aliases(con):
    """[(src, dst, src_tag)] for unreviewed POSSIBLE_ALIAS edges."""
    return con.execute(
        "SELECT src, dst, src_tag FROM edges WHERE rel='POSSIBLE_ALIAS' "
        "AND src_tag='alias-candidate' ORDER BY src").fetchall()


def pending_categories(con):
    """{label: [member, ...]} for unreviewed INSTANCE_OF proposals."""
    out = {}
    for src, dst in con.execute(
            "SELECT src, dst FROM edges WHERE rel='INSTANCE_OF' "
            "AND src_tag='proposed' ORDER BY dst, src"):
        out.setdefault(dst, []).append(src)
    return out


def approve_alias(con, loser, winner, hint_type=None):
    """Merge loser into winner: alias index for future writes + graph rewrite."""
    from entity_resolve import Resolver
    row = con.execute(
        "SELECT type FROM graph_nodes WHERE name=?", (winner,)).fetchone()
    ht = hint_type or (row[0] if row and row[0] not in ("", "concept") else None)
    rz = Resolver()
    rz.add_alias(loser, winner, ht)
    rz.save()
    counts = ms.merge_nodes(con, loser, winner)
    return {"status": "merged", "loser": loser, "winner": winner, **counts}


def reject_alias(con, a, b):
    """Durable no: keep the edge, tag rejected (PK suppresses re-proposal)."""
    n = con.execute(
        "UPDATE edges SET src_tag='rejected' WHERE rel='POSSIBLE_ALIAS' "
        "AND ((src=? AND dst=?) OR (src=? AND dst=?))", (a, b, b, a)).rowcount
    con.commit()
    return {"status": "rejected" if n else "not-found", "pair": [a, b]}


def approve_category(con, label):
    n = con.execute(
        "UPDATE edges SET src_tag='curated' WHERE rel='INSTANCE_OF' "
        "AND dst=? AND src_tag='proposed'", (label,)).rowcount
    con.commit()
    return {"status": "approved" if n else "not-found", "category": label, "members": n}


def reject_category(con, label):
    """Node kept (rejected edges reference it); load_graph filters both."""
    n = con.execute(
        "UPDATE edges SET src_tag='rejected' WHERE rel='INSTANCE_OF' "
        "AND dst=? AND src_tag='proposed'", (label,)).rowcount
    con.commit()
    return {"status": "rejected" if n else "not-found", "category": label, "members": n}


def summary(con):
    """Human-readable pending queue; empty string when nothing is pending."""
    aliases = pending_aliases(con)
    cats = pending_categories(con)
    if not aliases and not cats:
        return ""
    lines = ["Memory graph proposals awaiting your review:"]
    if aliases:
        lines.append(f"\nSame entity? ({len(aliases)} pairs)")
        for i, (a, b, _) in enumerate(aliases, 1):
            lines.append(f"  {i}. {a}  =  {b}")
    if cats:
        lines.append(f"\nProposed categories ({len(cats)})")
        for label, members in cats.items():
            shown = ", ".join(members[:6]) + (" ..." if len(members) > 6 else "")
            lines.append(f"  - {label} ({len(members)}): {shown}")
    lines.append("\nReply to approve/reject (e.g. 'approve alias 1, "
                 "reject the categories') and the agent applies it.")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    con = ms.connect()
    ms.init_schema(con)
    try:
        if args == ["--nudge"]:
            s = summary(con)
            if s:  # empty stdout = silent success = no cron delivery
                print(s)
            return 0
        if args == ["list"] or not args:
            print(summary(con) or "no pending proposals")
            return 0
        if len(args) >= 3 and args[0] in ("approve", "reject"):
            verb, kind = args[0], args[1]
            if kind == "alias" and len(args) == 4:
                fn = approve_alias if verb == "approve" else reject_alias
                print(fn(con, args[2], args[3]))
                return 0
            if kind == "category":
                fn = approve_category if verb == "approve" else reject_category
                print(fn(con, args[2]))
                return 0
        print(__doc__.split("CLI:")[1])
        return 2
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
