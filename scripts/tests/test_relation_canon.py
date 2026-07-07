"""Closed relation vocabulary added 2026-07-06 (canon_relation + load_graph
proposal filtering)."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import memstore as ms  # noqa: E402
from entity_resolve import canon_relation, normalize_relation  # noqa: E402


def test_exact_map_hits():
    assert canon_relation("works_at") == ("WORKS_FOR", False)
    assert canon_relation("HAS_JOB_POSTING") == ("HIRING", False)
    assert canon_relation("applied_for") == ("APPLIED_TO", False)
    assert canon_relation("IS_LOCATION") == ("LOCATED_IN", False)
    assert canon_relation("mentioned") == ("ABOUT", False)
    assert canon_relation("attended") == ("ATTENDS", False)


def test_flip_map_swaps_direction():
    assert canon_relation("OFFERED_BY") == ("OFFERS", True)
    assert canon_relation("posted_by") == ("HIRING", True)
    assert canon_relation("LED_BY") == ("LEADS", True)
    assert canon_relation("ROLE_AT") == ("HIRING", True)
    assert canon_relation("child_of") == ("PARENT_OF", True)
    assert canon_relation("employs") == ("WORKS_FOR", True)


def test_stem_rules_and_ordering():
    # location wins over OFFER even when both stems appear
    assert canon_relation("OFFERS_HYBRID_IN") == ("LOCATED_IN", False)
    assert canon_relation("LOCATION_OF_JOB_OPENING") == ("LOCATED_IN", False)
    # APPLI before JOB
    assert canon_relation("APPLICATION_SUBMITTED_TO") == ("APPLIED_TO", False)
    assert canon_relation("GTM_RECRUITING_LEAD_AT") == ("HIRING", False)
    assert canon_relation("COFOUNDER_DYNAMIC_WITH") == ("AFFILIATED_WITH", False)
    assert canon_relation("PREV_EMPLOYEE_OF") == ("WORKS_FOR", False)
    assert canon_relation("SENT_MONEY_VIA") == ("RELATED_TO", False)


def test_generic_passive_by_suffix_flips_stem_matches():
    assert canon_relation("IS_DEVELOPED_AND_OFFERED_BY") == ("OFFERS", True)
    assert canon_relation("CURRENTLY_MANAGED_BY") == ("LEADS", True)
    # unknown passive with no stem stays RELATED_TO unflipped
    assert canon_relation("BLESSED_BY") == ("RELATED_TO", False)


def test_unknown_falls_to_related_to_not_passthrough():
    # the pre-2026-07-06 sprawl engine returned QUACKS_LIKE here
    assert canon_relation("QUACKS_LIKE") == ("RELATED_TO", False)
    assert canon_relation("") == ("RELATED_TO", False)
    assert canon_relation(None) == ("RELATED_TO", False)


def test_system_relations_round_trip():
    assert canon_relation("INSTANCE_OF") == ("INSTANCE_OF", False)
    assert canon_relation("POSSIBLE_ALIAS") == ("POSSIBLE_ALIAS", False)


def test_normalize_relation_back_compat():
    assert normalize_relation("offered_by") == "OFFERS"
    assert normalize_relation("works_at") == "WORKS_FOR"


def _graph_db(tmp_path):
    con = ms.connect(tmp_path / "graph.db")
    ms.init_schema(con)
    ms.add_edge(con, "A", "WORKS_FOR", "B", "ingest")
    ms.add_edge(con, "A", "POSSIBLE_ALIAS", "A2", "alias-candidate")
    ms.add_edge(con, "C", "INSTANCE_OF", "cats", "proposed")
    ms.add_edge(con, "D", "INSTANCE_OF", "cats", "rejected")
    ms.add_edge(con, "E", "INSTANCE_OF", "cats", "curated")
    con.commit()
    return con


def test_load_graph_excludes_proposals_by_default(tmp_path):
    con = _graph_db(tmp_path)
    G = ms.load_graph(con)
    rels = {(u, v, d["relation"]) for u, v, d in G.edges(data=True)}
    assert ("A", "B", "WORKS_FOR") in rels
    assert ("E", "cats", "INSTANCE_OF") in rels          # curated survives
    assert ("A", "A2", "POSSIBLE_ALIAS") not in rels
    assert ("C", "cats", "INSTANCE_OF") not in rels      # proposed hidden
    assert ("D", "cats", "INSTANCE_OF") not in rels      # rejected hidden


def test_load_graph_escape_hatch(tmp_path):
    con = _graph_db(tmp_path)
    G = ms.load_graph(con, exclude_tags=None)
    rels = {(u, v, d["relation"]) for u, v, d in G.edges(data=True)}
    assert ("A", "A2", "POSSIBLE_ALIAS") in rels
    assert ("C", "cats", "INSTANCE_OF") in rels


def test_rel_orig_persisted(tmp_path):
    con = ms.connect(tmp_path / "ro.db")
    ms.init_schema(con)
    ms.add_edge(con, "X", "OFFERS", "Y", "ingest", rel_orig="OFFERED_BY")
    con.commit()
    row = con.execute("SELECT rel, rel_orig FROM edges WHERE src='X'").fetchone()
    assert row == ("OFFERS", "OFFERED_BY")
