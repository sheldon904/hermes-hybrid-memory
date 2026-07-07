"""One-time relation remap migration (2026-07-06)."""

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import memstore as ms  # noqa: E402

MIG = BASE / "migrations" / "2026-07-relation-remap.py"
spec = importlib.util.spec_from_file_location("relation_remap_under_test", MIG)
mig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mig)


def _db(tmp_path):
    con = ms.connect(tmp_path / "remap.db")
    ms.init_schema(con)
    # tail relation -> stem rule
    ms.add_edge(con, "the owner", "GTM_RECRUITING_LEAD_AT", "SomeCo", "ingest", source_ref="email:1")
    # inverse form -> flip
    ms.add_edge(con, "Widget", "OFFERED_BY", "Acme", "ingest", source_ref="email:2")
    # flip that collides with an existing canonical row (canonical wins provenance)
    ms.add_edge(con, "Acme", "OFFERS", "Gadget", "ingest", source_ref="email:old")
    ms.add_edge(con, "Gadget", "OFFERED_BY", "Acme", "ingest", source_ref="email:new")
    # already canonical -> untouched
    ms.add_edge(con, "the owner", "WORKS_FOR", "Acme Labs", "curated")
    # system rows -> untouched
    ms.add_edge(con, "A", "POSSIBLE_ALIAS", "B", "alias-candidate")
    ms.add_edge(con, "C", "INSTANCE_OF", "cats", "proposed")
    # idea nodes for retype
    ms.add_node(con, "Senior Rust Developer", "idea", {})
    ms.add_node(con, "Weekend camping trip", "idea", {})
    ms.add_node(con, "Mystery Role", "idea", {})
    ms.add_edge(con, "the owner", "APPLIED_TO", "Mystery Role", "ingest")
    con.commit()
    return con


def test_remap_flips_collides_and_preserves_provenance(tmp_path):
    con = _db(tmp_path)
    mapping, moved, dupes, selfs = mig.remap_relations(con)
    rows = {(r[0], r[1], r[2]): (r[3], r[4], r[5]) for r in con.execute(
        "SELECT src, rel, dst, src_tag, source_ref, rel_orig FROM edges")}
    # stem: GTM_RECRUITING_LEAD_AT -> HIRING, rel_orig kept
    assert rows[("the owner", "HIRING", "SomeCo")][1] == "email:1"
    assert rows[("the owner", "HIRING", "SomeCo")][2] == "GTM_RECRUITING_LEAD_AT"
    # flip: Widget OFFERED_BY Acme -> Acme OFFERS Widget
    assert ("Acme", "OFFERS", "Widget") in rows
    assert ("Widget", "OFFERED_BY", "Acme") not in rows
    # collision: existing canonical row keeps its (older) source_ref
    assert rows[("Acme", "OFFERS", "Gadget")][1] == "email:old"
    assert dupes == 1
    # untouched rows
    assert ("the owner", "WORKS_FOR", "Acme Labs") in rows
    assert ("A", "POSSIBLE_ALIAS", "B") in rows
    assert ("C", "INSTANCE_OF", "cats") in rows


def test_remap_idempotent(tmp_path):
    con = _db(tmp_path)
    mig.remap_relations(con)
    mapping, moved, dupes, selfs = mig.remap_relations(con)
    assert mapping == {} and moved == 0 and dupes == 0 and selfs == 0


def test_retype_nodes(tmp_path):
    con = _db(tmp_path)
    mig.remap_relations(con)
    retyped = mig.retype_nodes(con)
    assert "Senior Rust Developer" in retyped      # title regex
    assert "Mystery Role" in retyped               # APPLIED_TO incident edge
    assert "Weekend camping trip" not in retyped   # genuine idea untouched
    types = dict(con.execute(
        "SELECT name, type FROM graph_nodes WHERE name IN "
        "('Senior Rust Developer','Weekend camping trip','Mystery Role')"))
    assert types["Senior Rust Developer"] == "job"
    assert types["Mystery Role"] == "job"
    assert types["Weekend camping trip"] == "idea"
    # idempotent
    assert mig.retype_nodes(con) == []


def test_dry_run_writes_nothing(tmp_path):
    con = _db(tmp_path)
    before = con.execute("SELECT src, rel, dst FROM edges ORDER BY 1,2,3").fetchall()
    mapping, moved, dupes, selfs = mig.remap_relations(con, apply=False)
    assert mapping and moved == 0
    retyped = mig.retype_nodes(con, apply=False)
    assert retyped
    after = con.execute("SELECT src, rel, dst FROM edges ORDER BY 1,2,3").fetchall()
    assert before == after
    assert con.execute(
        "SELECT type FROM graph_nodes WHERE name='Mystery Role'").fetchone()[0] == "idea"
