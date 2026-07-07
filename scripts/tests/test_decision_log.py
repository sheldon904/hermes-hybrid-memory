"""Decision traces added 2026-07-06 (Phase 3): record/resolve tool,
feedback sweep, weekly report."""

import importlib.util
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import memstore as ms  # noqa: E402
import memory_feedback as mf  # noqa: E402
import memory_consolidate as mc  # noqa: E402

HYBRID = BASE / "plugins" / "hybrid" / "__init__.py"
spec = importlib.util.spec_from_file_location("hybrid_decision_under_test", HYBRID)
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)


def _fresh(tmp_path):
    con = ms.connect(tmp_path / "test_memory.db")
    ms.init_schema(con)
    # minimal facts table (owned by holographic in prod)
    con.execute("CREATE TABLE IF NOT EXISTS facts (fact_id INTEGER PRIMARY KEY, "
                "content TEXT, category TEXT, tags TEXT DEFAULT '', "
                "trust_score REAL DEFAULT 0.5, retrieval_count INTEGER DEFAULT 0, "
                "helpful_count INTEGER DEFAULT 0, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, source_ref TEXT DEFAULT '')")
    con.execute("INSERT INTO facts(fact_id, content, category) VALUES "
                "(42, 'Vendor X quoted 5k for the build', 'general')")
    con.commit()
    prov = object.__new__(hybrid.HybridMemoryProvider)
    prov._memcon = con
    prov._session_id = "s1"
    prov._turn_number = 5
    return con, prov


def test_record_links_turn_recalls(tmp_path):
    con, prov = _fresh(tmp_path)
    ms.log_recall(con, [("s1", 5, "vendor question", "vector", 42, "holo_fact_42", 0.6)])
    out = json.loads(prov._handle_decision_log(
        {"action": "record", "kind": "recommendation",
         "proposal": "Go with Vendor X", "options_shown": ["Vendor X", "Vendor Y"]}))
    assert out["status"] == "recorded" and out["linked_recalls"] == 1
    row = con.execute("SELECT kind, outcome, options_shown, source_refs "
                      "FROM decision_log WHERE id=?", (out["decision_id"],)).fetchone()
    assert row[0] == "recommendation" and row[1] == "pending"
    assert json.loads(row[2]) == ["Vendor X", "Vendor Y"]
    assert json.loads(row[3])[0]["fact_id"] == 42


def test_resolve_latest_pending_and_by_id(tmp_path):
    con, prov = _fresh(tmp_path)
    d1 = json.loads(prov._handle_decision_log(
        {"action": "record", "proposal": "First"}))["decision_id"]
    d2 = json.loads(prov._handle_decision_log(
        {"action": "record", "proposal": "Second"}))["decision_id"]
    # no id -> latest pending in session (d2)
    out = json.loads(prov._handle_decision_log(
        {"action": "resolve", "outcome": "accepted"}))
    assert out["decision_id"] == d2
    # explicit id with reason
    out = json.loads(prov._handle_decision_log(
        {"action": "resolve", "decision_id": d1, "outcome": "rejected",
         "reason": "too expensive"}))
    assert out["outcome"] == "rejected"
    rows = dict(con.execute("SELECT id, outcome FROM decision_log").fetchall())
    assert rows == {d1: "rejected", d2: "accepted"}
    # double-resolve is refused
    err = json.loads(prov._handle_decision_log(
        {"action": "resolve", "decision_id": d1, "outcome": "accepted"}))
    assert "error" in err


def test_sweep_boosts_accepted_skips_rejected_ages_pending(tmp_path):
    con, prov = _fresh(tmp_path)
    con.row_factory = __import__("sqlite3").Row
    refs = json.dumps([{"fact_id": 42, "vid": "holo_fact_42"}])
    con.execute("INSERT INTO decision_log(session_id, turn_number, kind, proposal, "
                "outcome, source_refs) VALUES ('s1', 5, 'recommendation', 'A', "
                "'accepted', ?)", (refs,))
    con.execute("INSERT INTO decision_log(session_id, turn_number, kind, proposal, "
                "outcome, source_refs) VALUES ('s1', 6, 'proposal', 'B', "
                "'rejected', ?)", (refs,))
    con.execute("INSERT INTO decision_log(session_id, turn_number, proposal, outcome, ts) "
                "VALUES ('s1', 7, 'C', 'pending', datetime('now', '-10 days'))")
    con.commit()
    n_ignored, pos, resolved_turns, applied = mf.sweep_decisions(con, apply=True)
    assert n_ignored == 1
    assert pos == {42: mf.POS_DELTA}
    assert ("s1", 5) in resolved_turns and ("s1", 6) in resolved_turns
    assert len(applied) == 2
    aged = con.execute("SELECT outcome FROM decision_log WHERE proposal='C'").fetchone()[0]
    assert aged == "ignored"


def test_decision_report_runs(tmp_path, capsys):
    con, prov = _fresh(tmp_path)
    prov._handle_decision_log({"action": "record", "proposal": "Use the managed host"})
    prov._handle_decision_log({"action": "resolve", "outcome": "accepted"})
    mc.decision_report(con)
    out = capsys.readouterr().out
    assert "Decision traces" in out and "acceptance 100%" in out
