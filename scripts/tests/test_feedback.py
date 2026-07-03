"""memory_feedback engagement heuristic, delta clamps, dry-run safety."""

import memory_feedback as mf
import memstore as ms
from conftest import add_fact


def _log(con, session, turn, query, fact_id, offset_days=0):
    con.execute(
        "INSERT INTO recall_log(ts, session_id, turn_number, query, block, fact_id, vid, score) "
        "VALUES (datetime('now', ?), ?, ?, ?, 'vector', ?, ?, 0.5)",
        (f"-{offset_days} days", session, turn, query, fact_id,
         f"holo_fact_{fact_id}" if fact_id else None))
    con.commit()


def test_engaged_fact_gets_boost(facts_db, monkeypatch):
    path, con = facts_db
    fid = add_fact(con, "Acme Construction is the sole paying client", entities=["Acme Construction"])
    _log(con, "s1", 1, "status of the platform", fid)
    _log(con, "s1", 2, "tell me more about Acme Construction billing", None)

    monkeypatch.setattr(mf, "DB", path)
    pos, neg = mf.compute_deltas(con)
    assert fid in pos and fid not in neg

    changed = mf.apply_deltas(con, pos, apply=True, reason="engaged")
    assert changed == 1
    trust, = con.execute("SELECT trust_score FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert abs(trust - 0.52) < 1e-6
    log = con.execute("SELECT old_trust, new_trust, reason FROM trust_log").fetchone()
    assert log[0] == 0.5 and abs(log[1] - 0.52) < 1e-6 and log[2] == "engaged"


def test_unengaged_needs_five_turns(facts_db):
    path, con = facts_db
    fid = add_fact(con, "totally unrelated trivia nobody asks about")
    for turn in range(1, 5):  # only 4 distinct turns -> no demotion
        _log(con, "s1", turn, f"question {turn}", fid, offset_days=2)
    pos, neg = mf.compute_deltas(con)
    assert fid not in neg
    _log(con, "s1", 5, "question 5", fid, offset_days=2)  # 5th turn
    pos, neg = mf.compute_deltas(con)
    assert fid in neg and fid not in pos


def test_clamps(facts_db):
    path, con = facts_db
    fid = add_fact(con, "already highly trusted fact")
    con.execute("UPDATE facts SET trust_score=0.845 WHERE fact_id=?", (fid,))
    mf.apply_deltas(con, {fid: mf.POS_DELTA}, apply=True, reason="engaged")
    trust, = con.execute("SELECT trust_score FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert trust == mf.TRUST_CEILING

    con.execute("UPDATE facts SET trust_score=0.36 WHERE fact_id=?", (fid,))
    mf.apply_deltas(con, {fid: mf.NEG_DELTA}, apply=True, reason="surfaced-unengaged")
    trust, = con.execute("SELECT trust_score FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert trust == mf.TRUST_FLOOR
    assert mf.TRUST_FLOOR > 0.3  # never crosses the recall threshold


def test_dry_run_writes_nothing(facts_db):
    path, con = facts_db
    fid = add_fact(con, "engaged fact", entities=["Widget Co"])
    _log(con, "s1", 1, "q", fid)
    _log(con, "s1", 2, "more about Widget Co please", None)
    pos, _ = mf.compute_deltas(con)
    mf.apply_deltas(con, pos, apply=False, reason="engaged")
    trust, = con.execute("SELECT trust_score FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert trust == 0.5
    assert con.execute("SELECT COUNT(*) FROM trust_log").fetchone()[0] == 0
