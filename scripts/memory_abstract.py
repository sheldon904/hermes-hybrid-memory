#!/usr/bin/env python3
"""memory_abstract.py, weekly abstraction pass over the fact store + graph.

Two Hofstadter mechanisms, both reversible and never destructive:

1. CHUNKING (concepts glom into larger units, internals stay recoverable):
   Clusters of related facts (union-find over shared entities) get summarized
   by flash-lite into ONE chunk fact (category='chunk') plus chunk_members
   rows and a graph 'episode' node. Members are NEVER deleted, the hybrid
   provider suppresses member lines when their chunk surfaces, and
   chunk_expand unpacks them on demand. Operational (email/lead) facts are
   INCLUDED: a chunk summarizing lead exhaust is how quarantined detail
   re-enters default recall in compressed form.

2. INSTANCE_OF PROMOTION (a single memory trace is a nascent category):
   Groups of same-type graph nodes whose relational fingerprints are nearly
   identical get a proposed category node and INSTANCE_OF edges tagged
   src_tag='proposed' (reviewable; wipe with remove_edges_by_tag).

Cluster rules: facts connect when they share >= 2 entities, or share 1 entity
that links to <= 15 facts (hub guard: the owner's own name connects almost
every fact and means nothing as a clustering signal). Components of 3-12
facts, every member >= 7 days old, no member already chunked. LLM failures
skip the cluster (retry next week).

Usage:
  memory_abstract.py            # DRY RUN, report clusters/groups only
  memory_abstract.py --apply    # write chunks + proposals
"""

import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "scripts"))
sys.path.insert(0, str(HERMES_HOME / "hermes-agent"))

import memstore as ms  # noqa: E402
import memory_ingest as mi  # noqa: E402

MIN_CLUSTER = 3
MAX_CLUSTER = 12
HUB_MAX_FACTS = 15         # an entity linked to more facts than this can't
                           # single-handedly connect a pair
PAIR_ENTITY_CAP = 50       # entities above this are skipped for pair generation
MIN_AGE_DAYS = 7
MAX_CHUNKS_PER_RUN = 6
FP_COS_MIN = 0.8
MAX_GROUP = 12
MAX_CATEGORY_GROUPS = 5
SKIP_NODE_TYPES = {"situation", "category", "episode"}


def _llm_json(system: str, user: str, max_tokens: int = 700):
    """One flash-lite call returning the first JSON object found, or None.
    Copies the memory_ingest.extract() contract: warn + None on any failure."""
    api_key = mi._read_env_key("OPENROUTER_API_KEY")
    if not api_key:
        print("  [skip] no OPENROUTER_API_KEY", file=sys.stderr)
        return None
    payload = json.dumps({
        "model": mi.INGEST_MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            content = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return None
        return json.loads(content[start:end + 1])
    except Exception as e:
        print(f"  [warn] LLM call failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def find_clusters(conn):
    """Union-find over shared entities. Returns list of (signature, [fact rows])."""
    eligible = {}
    for r in conn.execute(
            "SELECT fact_id, content, category, created_at FROM facts "
            "WHERE category != 'chunk' "
            "AND created_at <= datetime('now', ?) "
            "AND fact_id NOT IN (SELECT member_fact_id FROM chunk_members) "
            "AND fact_id NOT IN (SELECT chunk_fact_id FROM chunk_members)",
            (f"-{MIN_AGE_DAYS} days",)):
        eligible[r[0]] = {"fact_id": r[0], "content": r[1] or "",
                          "category": r[2] or "general", "created_at": r[3]}
    if not eligible:
        return []

    ent_facts = defaultdict(list)
    for fid, eid in conn.execute("SELECT fact_id, entity_id FROM fact_entities"):
        if fid in eligible:
            ent_facts[eid].append(fid)

    pair_shared = Counter()
    pair_small = set()
    for eid, fids in ent_facts.items():
        n = len(fids)
        if n < 2 or n > PAIR_ENTITY_CAP:
            continue
        small = n <= HUB_MAX_FACTS
        fids = sorted(fids)
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                pair = (fids[i], fids[j])
                pair_shared[pair] += 1
                if small:
                    pair_small.add(pair)

    uf = _UF()
    for pair, shared in pair_shared.items():
        if shared >= 2 or pair in pair_small:
            uf.union(pair[0], pair[1])

    comps = defaultdict(list)
    for fid in {f for pair in pair_shared for f in pair}:
        comps[uf.find(fid)].append(fid)

    clusters = []
    for members in comps.values():
        if not (MIN_CLUSTER <= len(members) <= MAX_CLUSTER):
            continue
        members = sorted(members)
        sig = hashlib.sha256(",".join(map(str, members)).encode()).hexdigest()[:12]
        used = conn.execute(
            "SELECT 1 FROM facts WHERE category='chunk' AND tags LIKE ?",
            (f"%sig:{sig}%",)).fetchone()
        if used:
            continue
        clusters.append((sig, [eligible[f] for f in members]))
    return clusters


def summarize_cluster(facts):
    """flash-lite -> (title, summary) or None."""
    listing = "\n".join(f"- {f['content'][:280]}" for f in facts)
    out = _llm_json(
        "You compress a cluster of related memory facts into one chunk. Output ONE "
        'JSON object: {"title": "...", "summary": "..."}. The title is a short '
        "episode/topic label (<= 60 chars). The summary is <= 350 chars, third "
        "person, self-contained, and preserves the load-bearing specifics "
        "(names, numbers, dates) while dropping repetition.",
        f"FACTS:\n{listing}")
    if not out or not isinstance(out.get("summary"), str):
        return None
    title = (out.get("title") or "").strip()[:60] or "Memory chunk"
    summary = out["summary"].strip()[:400]
    return title, summary


def cluster_entities(conn, member_ids, limit=6):
    qmarks = ",".join("?" * len(member_ids))
    rows = conn.execute(
        f"SELECT e.name, COUNT(*) c FROM fact_entities fe "
        f"JOIN entities e ON e.entity_id = fe.entity_id "
        f"WHERE fe.fact_id IN ({qmarks}) GROUP BY e.name "
        f"ORDER BY c DESC LIMIT ?", list(member_ids) + [limit]).fetchall()
    return [r[0] for r in rows if r[0]]


def write_chunk(store, mcon, sig, title, summary, member_ids, entities):
    fact_id = store.add_fact(summary, category="chunk", tags=f"chunk,sig:{sig}")
    for mid in member_ids:
        mcon.execute(
            "INSERT OR IGNORE INTO chunk_members(chunk_fact_id, member_fact_id) VALUES (?,?)",
            (fact_id, mid))
    ms.add_node(mcon, title, "episode", {"chunk_fact_id": fact_id})
    for ent in entities:
        ms.add_edge(mcon, title, "INVOLVES", ent, src_tag="chunk")
    mcon.commit()
    return fact_id


# ---------------------------------------------------------------------------
# INSTANCE_OF promotion
# ---------------------------------------------------------------------------

def _fingerprint(G, node):
    fp = Counter()
    for _, nb, d in G.out_edges(node, data=True):
        fp[("out", d.get("relation", "RELATED_TO"), G.nodes[nb].get("node_type", "concept"))] += 1
    for nb, _, d in G.in_edges(node, data=True):
        fp[("in", d.get("relation", "RELATED_TO"), G.nodes[nb].get("node_type", "concept"))] += 1
    return fp


def _cos(a, b):
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def propose_categories(mcon):
    """Groups of same-type nodes with near-identical relational fingerprints.
    Returns [(node_type, [member names], shared relation labels)]."""
    G = ms.load_graph(mcon)
    promoted = {r[0] for r in mcon.execute(
        "SELECT DISTINCT src FROM edges WHERE rel='INSTANCE_OF'")}
    by_type = defaultdict(list)
    for n in G.nodes():
        nt = G.nodes[n].get("node_type", "concept")
        if nt in SKIP_NODE_TYPES or n in promoted:
            continue
        if G.degree(n) < 2:
            continue
        fp = _fingerprint(G, n)
        if fp:
            by_type[nt].append((n, fp))

    groups = []
    for nt, items in by_type.items():
        if len(items) < 2:
            continue
        uf = _UF()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _cos(items[i][1], items[j][1]) >= FP_COS_MIN:
                    uf.union(items[i][0], items[j][0])
        comps = defaultdict(list)
        fps = dict(items)
        for n, _ in items:
            comps[uf.find(n)].append(n)
        for members in comps.values():
            if len(members) < 2 or len(members) > MAX_GROUP:
                continue  # singletons and giant blobs (e.g. all leads) skipped
            shared = set(fps[members[0]])
            for m in members[1:]:
                shared &= set(fps[m])
            rels = sorted({rel for (_, rel, _) in shared})
            groups.append((nt, sorted(members), rels))
    groups.sort(key=lambda g: -len(g[1]))
    return groups[:MAX_CATEGORY_GROUPS]


def name_category(node_type, members, rels):
    out = _llm_json(
        "You name an emergent category. Given member entities (all the same type) "
        "and the relationship kinds they share, output ONE JSON object: "
        '{"label": "..."}, a short lowercase category label (<= 40 chars) that '
        "captures what the members have in common. No punctuation except spaces/hyphens.",
        f"TYPE: {node_type}\nMEMBERS: {', '.join(members[:12])}\n"
        f"SHARED RELATIONS: {', '.join(rels) or '(none)'}")
    if not out or not isinstance(out.get("label"), str):
        return None
    label = out["label"].strip().lower()[:40]
    return label or None


def write_proposal(mcon, label, members):
    ms.add_node(mcon, label, "category", {"proposed_by": "memory_abstract"})
    for m in members:
        ms.add_edge(mcon, m, "INSTANCE_OF", label, src_tag="proposed")
    mcon.commit()


# ---------------------------------------------------------------------------

def main():
    apply = "--apply" in sys.argv
    skip_chunks = "--skip-chunks" in sys.argv
    skip_categories = "--skip-categories" in sys.argv

    mcon = ms.connect()
    ms.init_schema(mcon)

    store = None
    conn = None
    try:
        from plugins.memory.holographic.store import MemoryStore
        store = MemoryStore(db_path=str(HERMES_HOME / "memory_store.db"))
        conn = store._conn
    except Exception as e:
        print(f"[warn] holographic store unavailable ({e}); chunking disabled",
              file=sys.stderr)
        skip_chunks = True

    wrote_chunks = 0
    if not skip_chunks:
        clusters = find_clusters(conn)
        print(f"chunk clusters found: {len(clusters)} (writing at most {MAX_CHUNKS_PER_RUN})")
        for sig, facts in clusters[:MAX_CHUNKS_PER_RUN]:
            member_ids = [f["fact_id"] for f in facts]
            print(f"  cluster {sig} ({len(facts)} facts):")
            for f in facts:
                print(f"    #{f['fact_id']} [{f['category']}] {f['content'][:60]!r}")
            if not apply:
                continue
            got = summarize_cluster(facts)
            if not got:
                print("    [skip] summarize failed")
                continue
            title, summary = got
            ents = cluster_entities(conn, member_ids)
            fid = write_chunk(store, mcon, sig, title, summary, member_ids, ents)
            wrote_chunks += 1
            print(f"    -> chunk #{fid} '{title}' involving {', '.join(ents[:4])}")

    wrote_groups = 0
    if not skip_categories:
        groups = propose_categories(mcon)
        print(f"category groups found: {len(groups)}")
        for nt, members, rels in groups:
            print(f"  [{nt}] {len(members)} members sharing {rels}: {', '.join(members[:6])}"
                  + (" ..." if len(members) > 6 else ""))
            if not apply:
                continue
            label = name_category(nt, members, rels)
            if not label:
                print("    [skip] naming failed")
                continue
            write_proposal(mcon, label, members)
            wrote_groups += 1
            print(f"    -> category '{label}' (INSTANCE_OF x{len(members)}, src_tag=proposed)")

    if apply:
        print(f"applied: {wrote_chunks} chunks, {wrote_groups} category proposals")
    else:
        print("\nDRY RUN, nothing written. Re-run with --apply.")

    if store:
        store.close()
    mcon.close()


if __name__ == "__main__":
    main()
