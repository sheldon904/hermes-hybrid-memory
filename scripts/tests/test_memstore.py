"""memstore schema + vector + recall_log behavior (no embedder needed)."""

import memstore as ms


def test_schema_idempotent(mem_db):
    ms.init_schema(mem_db)  # second run must not raise
    tables = {r[0] for r in mem_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("vec_items", "edges", "graph_nodes", "graph_extracted",
              "recall_log", "trust_log", "fact_gists", "gist_extracted",
              "chunk_members"):
        assert t in tables, f"missing table {t}"


def test_upsert_query_returns_vid(mem_db):
    emb = [0.1] * ms.DIM
    ms.upsert_vector(mem_db, "holo_fact_42", "the answer", "test", "note", embedding=emb)
    mem_db.commit()
    hits = ms.query_vectors(mem_db, embedding=emb, k=3)
    assert hits and hits[0]["vid"] == "holo_fact_42"
    assert hits[0]["text"] == "the answer"
    assert hits[0]["similarity"] > 0.99


def test_upsert_replaces(mem_db):
    emb = [0.2] * ms.DIM
    ms.upsert_vector(mem_db, "v1", "old text", "t", "note", embedding=emb)
    ms.upsert_vector(mem_db, "v1", "new text", "t", "note", embedding=emb)
    mem_db.commit()
    assert mem_db.execute("SELECT COUNT(*) FROM vec_items WHERE vid='v1'").fetchone()[0] == 1
    assert ms.query_vectors(mem_db, embedding=emb, k=1)[0]["text"] == "new text"


def test_log_recall(mem_db):
    ms.log_recall(mem_db, [("s1", 3, "q", "vector", 42, "holo_fact_42", 0.5),
                           ("s1", 3, "q", "graph", None, None, None)])
    rows = mem_db.execute(
        "SELECT session_id, turn_number, block, fact_id FROM recall_log ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0][:4] == ("s1", 3, "vector", 42)
    ms.log_recall(mem_db, [])  # empty is a no-op, not an error


def test_edges_and_graph(mem_db):
    ms.add_node(mem_db, "A", "company", {"stage": "client"})
    ms.add_node(mem_db, "A")  # attr-less upsert must not clobber
    ms.add_edge(mem_db, "A", "CLIENT_OF", "B", src_tag="test")
    mem_db.commit()
    G = ms.load_graph(mem_db)
    assert G.nodes["A"]["stage"] == "client"
    assert G.edges["A", "B"]["relation"] == "CLIENT_OF"
    ms.remove_edges_by_tag(mem_db, "test")
    assert mem_db.execute("SELECT COUNT(*) FROM edges WHERE src_tag='test'").fetchone()[0] == 0
