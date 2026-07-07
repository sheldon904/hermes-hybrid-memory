"""Operator review loop for graph proposals, added 2026-07-06:
merge_nodes, durable rejection (src_tag convention), plugin tool."""

import importlib.util
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import memstore as ms  # noqa: E402
import ontology_review as orv  # noqa: E402

HYBRID = BASE / "plugins" / "hybrid" / "__init__.py"
spec = importlib.util.spec_from_file_location("hybrid_ontology_under_test", HYBRID)
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)


def _db(tmp_path):
    con = ms.connect(tmp_path / "graph.db")
    ms.init_schema(con)
    return con


def test_merge_nodes_rewrites_dedupes_and_merges_attrs(tmp_path):
    con = _db(tmp_path)
    ms.add_node(con, "Acme Corp", "company", {"domain": "acme.com"})
    ms.add_node(con, "Acme Corporation", "company", {"stage": "prospect"})
    ms.add_edge(con, "Acme Corporation", "LOCATED_IN", "Tampa", "ingest", source_ref="email:1")
    ms.add_edge(con, "Acme Corp", "LOCATED_IN", "Tampa", "ingest", source_ref="email:2")  # will collide
    ms.add_edge(con, "the owner", "CONTACTED", "Acme Corporation", "ingest")
    ms.add_edge(con, "Acme Corporation", "POSSIBLE_ALIAS", "Acme Corp", "alias-candidate")
    con.commit()
    counts = ms.merge_nodes(con, "Acme Corporation", "Acme Corp")
    assert counts["moved"] >= 1 and counts["dropped"] >= 1
    rows = con.execute("SELECT src, rel, dst, source_ref FROM edges ORDER BY rel").fetchall()
    assert ("Acme Corp", "LOCATED_IN", "Tampa", "email:2") in rows  # existing row won (first-assertion)
    assert ("the owner", "CONTACTED", "Acme Corp", "") in rows
    assert not any(r[1] == "POSSIBLE_ALIAS" for r in rows)
    assert con.execute("SELECT COUNT(*) FROM graph_nodes WHERE name='Acme Corporation'").fetchone()[0] == 0
    attrs = json.loads(con.execute(
        "SELECT attrs FROM graph_nodes WHERE name='Acme Corp'").fetchone()[0])
    assert attrs == {"stage": "prospect", "domain": "acme.com"}  # winner wins, loser fills


def test_reject_alias_is_durable(tmp_path):
    con = _db(tmp_path)
    ms.add_edge(con, "Bob Smith", "POSSIBLE_ALIAS", "Rob Smith", "alias-candidate")
    con.commit()
    out = orv.reject_alias(con, "Rob Smith", "Bob Smith")  # reversed order still matches
    assert out["status"] == "rejected"
    assert orv.pending_aliases(con) == []
    # re-proposal from a later ingest is a PK no-op, tag stays 'rejected'
    ms.add_edge(con, "Bob Smith", "POSSIBLE_ALIAS", "Rob Smith", "alias-candidate")
    con.commit()
    assert orv.pending_aliases(con) == []
    tag = con.execute("SELECT src_tag FROM edges WHERE rel='POSSIBLE_ALIAS'").fetchone()[0]
    assert tag == "rejected"


def test_category_approve_and_reject(tmp_path):
    con = _db(tmp_path)
    for m in ("Boeing", "AbbVie"):
        ms.add_edge(con, m, "INSTANCE_OF", "large corps", "proposed")
    for m in ("Tampa", "Prague"):
        ms.add_edge(con, m, "INSTANCE_OF", "cities", "proposed")
    con.commit()
    assert set(orv.pending_categories(con)) == {"large corps", "cities"}
    assert orv.approve_category(con, "large corps")["members"] == 2
    assert orv.reject_category(con, "cities")["members"] == 2
    assert orv.pending_categories(con) == {}
    tags = dict(con.execute(
        "SELECT dst, src_tag FROM edges WHERE rel='INSTANCE_OF' GROUP BY dst, src_tag"))
    assert tags == {"large corps": "curated", "cities": "rejected"}
    # approved visible to readers, rejected hidden
    G = ms.load_graph(con)
    rels = {(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "INSTANCE_OF"}
    assert ("Boeing", "large corps") in rels and ("Tampa", "cities") not in rels


def test_approve_alias_updates_resolver_index(tmp_path, monkeypatch):
    con = _db(tmp_path)
    ms.add_node(con, "Acme Construction", "company", {})
    ms.add_edge(con, "ACME Construction LLC", "POSSIBLE_ALIAS", "Acme Construction", "alias-candidate")
    con.commit()
    saved = {}
    import entity_resolve as er

    class FakeResolver:
        def __init__(self):
            self.pending_aliases = []

        def add_alias(self, variant, canonical, hint_type=None):
            saved["alias"] = (variant, canonical, hint_type)

        def save(self):
            saved["saved"] = True

    monkeypatch.setattr(er, "Resolver", FakeResolver)
    out = orv.approve_alias(con, "ACME Construction LLC", "Acme Construction")
    assert out["status"] == "merged"
    assert saved["alias"][0] == "ACME Construction LLC"
    assert saved["alias"][1] == "Acme Construction"
    assert saved["alias"][2] == "company"  # hint from winner's node type
    assert saved.get("saved") is True
    assert orv.pending_aliases(con) == []


def test_plugin_tool_round_trip(tmp_path):
    con = _db(tmp_path)
    ms.add_edge(con, "X Co", "POSSIBLE_ALIAS", "X Company", "alias-candidate")
    ms.add_edge(con, "Y", "INSTANCE_OF", "things", "proposed")
    con.commit()
    prov = object.__new__(hybrid.HybridMemoryProvider)
    prov._memcon = con
    prov._load_graph = lambda: None
    out = json.loads(prov._handle_ontology_review({"action": "list"}))
    assert out["aliases"] == [{"a": "X Co", "b": "X Company"}]
    assert out["categories"] == [{"label": "things", "members": ["Y"]}]
    out = json.loads(prov._handle_ontology_review(
        {"action": "reject", "kind": "alias", "a": "X Co", "b": "X Company"}))
    assert out["status"] == "rejected"
    out = json.loads(prov._handle_ontology_review(
        {"action": "reject", "kind": "category", "label": "things"}))
    assert out["status"] == "rejected"
    out = json.loads(prov._handle_ontology_review({"action": "list"}))
    assert out["aliases"] == [] and out["categories"] == []


def test_nudge_summary_silent_when_empty(tmp_path):
    con = _db(tmp_path)
    assert orv.summary(con) == ""
    ms.add_edge(con, "A", "POSSIBLE_ALIAS", "B", "alias-candidate")
    con.commit()
    s = orv.summary(con)
    assert "A  =  B" in s and "review" in s.lower()
