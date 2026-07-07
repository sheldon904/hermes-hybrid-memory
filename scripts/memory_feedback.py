#!/usr/bin/env python3
"""memory_feedback.py, nightly usage-based trust adjustment (heuristic v1).

Closes the feedback loop the fact store was designed around: facts that get
surfaced AND engaged with gain trust; facts that keep getting surfaced but
never matter lose a little. Signals come entirely from recall_log (written by
the hybrid provider), no session-DB parsing, no LLM, no embedder.

Engagement heuristic: a surfaced fact is "engaged" if any LATER query in the
same session (a) mentions one of the fact's linked entities (name length >= 4),
or (b) overlaps the fact's content tokens with Jaccard >= 0.3. The user
returning to the fact's subject after it was surfaced is the cheapest honest
proxy for "that memory mattered".

Deltas (bounded, auditable, reversible via trust_log):
  +0.02 per run for facts engaged in the last --pos-days (ceiling 0.85)
  -0.02 for facts surfaced on >= 5 distinct turns in the last 14 days with
        ZERO engagement anywhere in that window (floor 0.35, deliberately
        above min_trust_threshold 0.3, so demoted facts never vanish from
        recall; they just rank lower)

Every change is recorded in trust_log(old_trust, new_trust, reason). Rows in
recall_log older than 60 days are pruned (with --apply).

Usage:
  memory_feedback.py             # DRY RUN, report only
  memory_feedback.py --apply     # write trust changes + prune recall_log
  memory_feedback.py --llm       # reserved for LLM-judged engagement (v2)
"""

import os
import sqlite3
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
DB = HERMES_HOME / "memory_store.db"

POS_DELTA = 0.02
NEG_DELTA = -0.02
TRUST_CEILING = 0.85
TRUST_FLOOR = 0.35
NEG_MIN_TURNS = 5
NEG_WINDOW_DAYS = 14
RECALL_LOG_RETENTION_DAYS = 60
JACCARD_ENGAGED = 0.3


def _tokens(s):
    return {t.strip(".,!?;:\"'()[]{}").lower() for t in (s or "").split() if len(t) > 2}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_surfacings(con, days):
    """recall_log rows with a fact_id from the last `days` days, plus the
    per-session ordered query stream for the engagement lookups."""
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, ts, session_id, turn_number, query, fact_id FROM recall_log "
        "WHERE fact_id IS NOT NULL AND ts >= datetime('now', ?) ORDER BY id",
        (f"-{int(days)} days",)).fetchall()
    stream = {}
    for r in con.execute(
            "SELECT id, session_id, query FROM recall_log "
            "WHERE ts >= datetime('now', ?) ORDER BY id",
            (f"-{int(days)} days",)):
        stream.setdefault(r["session_id"] or "", []).append((r["id"], r["query"] or ""))
    return rows, stream


def fact_entities(con, fact_id):
    try:
        return [r[0] for r in con.execute(
            "SELECT e.name FROM fact_entities fe "
            "JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id=?",
            (fact_id,)) if r[0] and len(r[0]) >= 4]
    except Exception:
        return []


def is_engaged(fact_row, entities, later_queries):
    """True if any later query mentions a linked entity or overlaps content."""
    content_toks = _tokens(fact_row["content"])
    ents = [e.lower() for e in entities]
    for q in later_queries:
        ql = q.lower()
        if any(e in ql for e in ents):
            return True
        if _jaccard(content_toks, _tokens(q)) >= JACCARD_ENGAGED:
            return True
    return False


DECISION_IGNORE_DAYS = 7


def sweep_decisions(con, apply):
    """Decision-trace pass (Phase 3, 2026-07-06). Ages pending decisions to
    'ignored' after DECISION_IGNORE_DAYS, and turns resolved decisions into
    trust signal: accepted/modified -> POS_DELTA for the facts recalled on the
    recommendation turn (source_refs). Rejected -> no boost. Each decision is
    applied to trust exactly once (trust_applied flag). Returns
    (n_ignored, pos_deltas, resolved_turns, applied_ids): resolved_turns are
    (session_id, turn_number) pairs whose surfacings the token-overlap
    heuristic should skip, the decision outcome is the better signal."""
    import json as _json
    try:
        con.execute("SELECT 1 FROM decision_log LIMIT 1")
    except sqlite3.OperationalError:
        return 0, {}, set(), []

    n_ignored = con.execute(
        "SELECT COUNT(*) FROM decision_log WHERE outcome='pending' "
        "AND ts < datetime('now', ?)", (f"-{DECISION_IGNORE_DAYS} days",)).fetchone()[0]
    if apply and n_ignored:
        con.execute(
            "UPDATE decision_log SET outcome='ignored' WHERE outcome='pending' "
            "AND ts < datetime('now', ?)", (f"-{DECISION_IGNORE_DAYS} days",))

    pos, resolved_turns, applied = {}, set(), []
    for r in con.execute(
            "SELECT id, session_id, turn_number, outcome, source_refs FROM decision_log "
            "WHERE outcome IN ('accepted','rejected','modified') AND trust_applied=0"):
        resolved_turns.add((r["session_id"], r["turn_number"]))
        applied.append(r["id"])
        if r["outcome"] == "rejected":
            continue
        try:
            refs = _json.loads(r["source_refs"] or "[]")
        except Exception:
            refs = []
        for ref in refs:
            fid = ref.get("fact_id") if isinstance(ref, dict) else None
            if fid:
                pos[int(fid)] = POS_DELTA
    return n_ignored, pos, resolved_turns, applied


def compute_deltas(con, pos_days=1, excluded_turns=None):
    con.row_factory = sqlite3.Row
    excluded_turns = excluded_turns or set()
    surf, stream = load_surfacings(con, NEG_WINDOW_DAYS)
    surf = [r for r in surf
            if (r["session_id"], r["turn_number"]) not in excluded_turns]
    if not surf:
        return {}, {}

    fact_rows = {}
    for r in surf:
        fact_rows.setdefault(r["fact_id"], []).append(r)

    facts = {}
    for fid in fact_rows:
        row = con.execute(
            "SELECT fact_id, content, category, trust_score FROM facts WHERE fact_id=?",
            (fid,)).fetchone()
        if row:
            facts[fid] = row

    engaged_recent, engaged_window, turns_in_window = {}, set(), {}
    ent_cache = {}
    for fid, rows in fact_rows.items():
        if fid not in facts:
            continue
        ent_cache.setdefault(fid, fact_entities(con, fid))
        turns = set()
        for r in rows:
            turns.add((r["session_id"], r["turn_number"]))
            later = [q for (rid, q) in stream.get(r["session_id"] or "", []) if rid > r["id"]]
            if not later:
                continue
            if is_engaged(facts[fid], ent_cache[fid], later):
                engaged_window.add(fid)
                recent = con.execute(
                    "SELECT 1 FROM recall_log WHERE id=? AND ts >= datetime('now', ?)",
                    (r["id"], f"-{int(pos_days)} days")).fetchone()
                if recent:
                    engaged_recent[fid] = True
        turns_in_window[fid] = len(turns)

    pos, neg = {}, {}
    for fid in engaged_recent:
        pos[fid] = POS_DELTA
    for fid, nturns in turns_in_window.items():
        if fid in engaged_window or fid in pos:
            continue
        if nturns >= NEG_MIN_TURNS:
            neg[fid] = NEG_DELTA
    return pos, neg


def apply_deltas(con, deltas, apply, reason):
    con.row_factory = sqlite3.Row
    changed = 0
    for fid, delta in sorted(deltas.items()):
        row = con.execute(
            "SELECT trust_score, content FROM facts WHERE fact_id=?", (fid,)).fetchone()
        if not row:
            continue
        old = float(row["trust_score"] or 0.5)
        new = old + delta
        new = min(new, TRUST_CEILING) if delta > 0 else max(new, TRUST_FLOOR)
        new = round(new, 4)
        if new == old:
            continue
        changed += 1
        print(f"  #{fid} {old:.2f} -> {new:.2f} [{reason}] {(row['content'] or '')[:60]!r}")
        if apply:
            con.execute(
                "INSERT INTO trust_log(fact_id, old_trust, new_trust, reason) VALUES (?,?,?,?)",
                (fid, old, new, reason))
            con.execute(
                "UPDATE facts SET trust_score=?, updated_at=CURRENT_TIMESTAMP WHERE fact_id=?",
                (new, fid))
    return changed


def main():
    apply = "--apply" in sys.argv
    if "--llm" in sys.argv:
        print("[note] --llm engagement judging is reserved for v2; running heuristic")
    if not DB.exists():
        print("no memory_store.db, nothing to do")
        return

    con = sqlite3.connect(str(DB), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")

    total = con.execute("SELECT COUNT(*) FROM recall_log").fetchone()[0]
    n_ignored, dec_pos, resolved_turns, applied_ids = sweep_decisions(con, apply)
    pos, neg = compute_deltas(con, excluded_turns=resolved_turns)
    # A decision outcome outranks the token-overlap proxy for the same fact.
    for fid in dec_pos:
        pos.pop(fid, None)
        neg.pop(fid, None)
    print(f"recall_log rows: {total} | engaged (recent): {len(pos)} | "
          f"surfaced-but-never-engaged (14d, >={NEG_MIN_TURNS} turns): {len(neg)} | "
          f"decisions: {len(applied_ids)} resolved, {n_ignored} aged to ignored")
    n_dec = apply_deltas(con, dec_pos, apply, "decision-accepted")
    n_pos = apply_deltas(con, pos, apply, "engaged")
    n_neg = apply_deltas(con, neg, apply, "surfaced-unengaged")

    if apply and applied_ids:
        con.executemany("UPDATE decision_log SET trust_applied=1 WHERE id=?",
                        [(i,) for i in applied_ids])

    pruned = 0
    if apply:
        cur = con.execute(
            "DELETE FROM recall_log WHERE ts < datetime('now', ?)",
            (f"-{RECALL_LOG_RETENTION_DAYS} days",))
        pruned = cur.rowcount
        con.commit()

    if apply:
        print(f"applied: +{n_dec} decision boosts, +{n_pos} boosts, {n_neg} demotions | "
              f"pruned {pruned} old recall_log rows")
    else:
        print(f"\nDRY RUN, nothing changed ({n_pos} boosts, {n_neg} demotions pending). "
              "Re-run with --apply to write.")
    con.close()


if __name__ == "__main__":
    main()
