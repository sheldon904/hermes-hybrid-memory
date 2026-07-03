#!/usr/bin/env python3
"""memstore.py — the unified Hermes memory store (one SQLite database).

Collapses what used to be three stores into memory_store.db:
  • facts + FTS5 + trust  — owned by the holographic provider (untouched here)
  • vectors               — sqlite-vec (cosine), this module
  • graph (edges/nodes)   — plain tables, this module, written incrementally

All additive: these tables do not collide with holographic's (facts/entities/
fact_entities/FTS). Writes are incremental and transactional; there is no
periodic full rebuild. networkx is loaded from the edges table on demand for
path/reason queries.

Embedder: ChromaDB's ONNX all-MiniLM-L6-v2 function (the model only — the
ChromaDB *store* is retired). 384-dim.
"""

import json
import os
import struct
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
DB_PATH = HERMES_HOME / "memory_store.db"
DIM = 384

_EF = None


def _ef():
    global _EF
    if _EF is None:
        from chromadb.utils import embedding_functions as ef
        _EF = ef.DefaultEmbeddingFunction()  # ONNX all-MiniLM-L6-v2
    return _EF


def embed(texts):
    return [[float(x) for x in v] for v in _ef()(list(texts))]


def _pack(v):
    return struct.pack("%df" % len(v), *v)


def connect(db_path=None):
    import sqlite3
    import sqlite_vec
    con = sqlite3.connect(str(db_path or DB_PATH), timeout=30, check_same_thread=False)
    con.execute("PRAGMA busy_timeout=30000")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def init_schema(con):
    con.execute("CREATE TABLE IF NOT EXISTS vec_items ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, vid TEXT UNIQUE, "
                "text TEXT, source TEXT, type TEXT)")
    con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0("
                f"item_id INTEGER PRIMARY KEY, embedding float[{DIM}] distance_metric=cosine)")
    con.execute("CREATE TABLE IF NOT EXISTS edges ("
                "src TEXT, rel TEXT, dst TEXT, src_tag TEXT, "
                "PRIMARY KEY(src, rel, dst))")
    con.execute("CREATE TABLE IF NOT EXISTS graph_nodes ("
                "name TEXT PRIMARY KEY, type TEXT, attrs TEXT)")
    # Watermark of facts already mined for graph edges (so extraction is incremental).
    con.execute("CREATE TABLE IF NOT EXISTS graph_extracted (fact_id INTEGER PRIMARY KEY)")
    # Every memory the provider surfaces (prefetch or tool), for the nightly
    # feedback pass. Pruned at 60 days by memory_feedback.py.
    con.execute("CREATE TABLE IF NOT EXISTS recall_log ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "session_id TEXT, turn_number INTEGER, query TEXT, "
                "block TEXT, fact_id INTEGER, vid TEXT, score REAL)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_recall_ts ON recall_log(ts)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_recall_fact ON recall_log(fact_id)")
    # Audit trail for every trust_score change made outside the live agent.
    con.execute("CREATE TABLE IF NOT EXISTS trust_log ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "fact_id INTEGER, old_trust REAL, new_trust REAL, reason TEXT)")
    # One-line "what this fact is an instance of" per fact; embedded as
    # vid gist_<fact_id> so recall can match on essence, not just surface.
    con.execute("CREATE TABLE IF NOT EXISTS fact_gists ("
                "fact_id INTEGER PRIMARY KEY, gist TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    con.execute("CREATE TABLE IF NOT EXISTS gist_extracted (fact_id INTEGER PRIMARY KEY)")
    # Chunk (category='chunk') facts and the member facts they summarize.
    con.execute("CREATE TABLE IF NOT EXISTS chunk_members ("
                "chunk_fact_id INTEGER, member_fact_id INTEGER, "
                "PRIMARY KEY(chunk_fact_id, member_fact_id))")
    con.commit()


# -- vectors -----------------------------------------------------------------

def upsert_vector(con, vid, text, source="", vtype="", embedding=None):
    if not text:
        return
    if embedding is None:
        embedding = embed([text])[0]
    row = con.execute("SELECT id FROM vec_items WHERE vid=?", (vid,)).fetchone()
    if row:
        iid = row[0]
        con.execute("UPDATE vec_items SET text=?, source=?, type=? WHERE id=?",
                    (text, source, vtype, iid))
        con.execute("DELETE FROM vec_index WHERE item_id=?", (iid,))
    else:
        con.execute("INSERT INTO vec_items(vid, text, source, type) VALUES (?,?,?,?)",
                    (vid, text, source, vtype))
        iid = con.execute("SELECT id FROM vec_items WHERE vid=?", (vid,)).fetchone()[0]
    con.execute("INSERT INTO vec_index(item_id, embedding) VALUES (?, ?)", (iid, _pack(embedding)))


def delete_vector(con, vid):
    row = con.execute("SELECT id FROM vec_items WHERE vid=?", (vid,)).fetchone()
    if row:
        con.execute("DELETE FROM vec_index WHERE item_id=?", (row[0],))
        con.execute("DELETE FROM vec_items WHERE id=?", (row[0],))


def query_vectors(con, query_text=None, embedding=None, k=5):
    if embedding is None:
        if not query_text:
            return []
        embedding = embed([query_text])[0]
    rows = con.execute(
        "SELECT vi.text, vi.source, vi.type, kk.distance, vi.vid FROM "
        "(SELECT item_id, distance FROM vec_index WHERE embedding MATCH ? AND k = ?) kk "
        "JOIN vec_items vi ON vi.id = kk.item_id ORDER BY kk.distance",
        (_pack(embedding), int(k))).fetchall()
    return [{"text": r[0], "source": r[1], "type": r[2],
             "similarity": round(1.0 - float(r[3]), 3), "vid": r[4]} for r in rows]


def vector_vids(con):
    return {r[0] for r in con.execute("SELECT vid FROM vec_items")}


def log_recall(con, rows):
    """Append surfacing events. rows: iterable of
    (session_id, turn_number, query, block, fact_id, vid, score)."""
    rows = list(rows)
    if not rows:
        return
    con.executemany(
        "INSERT INTO recall_log(session_id, turn_number, query, block, fact_id, vid, score) "
        "VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()


# -- graph -------------------------------------------------------------------

def add_node(con, name, ntype="concept", attrs=None):
    if not name:
        return
    attrs = attrs if isinstance(attrs, dict) else {}
    if attrs:
        # Real attributes → set them (newer wins).
        con.execute(
            "INSERT INTO graph_nodes(name, type, attrs) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "type=COALESCE(excluded.type, graph_nodes.type), attrs=excluded.attrs",
            (name, ntype, json.dumps(attrs, ensure_ascii=False)))
    else:
        # No attrs to contribute → insert if new, but NEVER clobber an existing
        # node's attrs/type (this is what was wiping lead domain/stage/score).
        con.execute(
            "INSERT INTO graph_nodes(name, type, attrs) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET type=COALESCE(graph_nodes.type, excluded.type)",
            (name, ntype, "{}"))


def add_edge(con, src, rel, dst, src_tag="ingest"):
    if not src or not dst or src == dst:
        return
    con.execute("INSERT OR IGNORE INTO edges(src, rel, dst, src_tag) VALUES (?,?,?,?)",
                (src, rel, dst, src_tag))


def remove_edges_by_tag(con, src_tag):
    con.execute("DELETE FROM edges WHERE src_tag=?", (src_tag,))


def load_graph(con):
    """Build a networkx DiGraph from the edges + graph_nodes tables."""
    import networkx as nx
    G = nx.DiGraph()
    for name, ntype, attrs in con.execute("SELECT name, type, attrs FROM graph_nodes"):
        a = {}
        try:
            a = json.loads(attrs) if attrs else {}
        except Exception:
            a = {}
        G.add_node(name, node_type=ntype or "concept",
                   **{k: v for k, v in a.items() if isinstance(v, (str, int, float, bool))})
    for src, rel, dst, tag in con.execute("SELECT src, rel, dst, src_tag FROM edges"):
        if not G.has_node(src):
            G.add_node(src, node_type="concept")
        if not G.has_node(dst):
            G.add_node(dst, node_type="concept")
        G.add_edge(src, dst, relation=rel, source=tag)
    return G


def stats(con):
    return {
        "vectors": con.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0],
        "edges": con.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "nodes": con.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0],
    }


if __name__ == "__main__":
    con = connect()
    init_schema(con)
    print("memstore schema ready at", DB_PATH)
    print("stats:", stats(con))
