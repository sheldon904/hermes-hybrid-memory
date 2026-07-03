#!/usr/bin/env python3
"""memstore_sync.py — incremental vectorization into the unified store.

Vectorizes any fact / memory entry not yet present in sqlite-vec, and prunes
vectors whose source was deleted. Only NEW items get embedded, so this is cheap.

Replaces the old memory_import.py (ChromaDB) and the 30-min build_kg rebuild:
  • vectors  — maintained here, incrementally
  • graph    — written incrementally at ingest time (memory_ingest / world_model_import)
  • facts    — owned by the holographic provider

Sources covered: holographic facts (memory_store.db), MEMORY.md, USER.md,
fact_store.json (job applications + metrics).
"""

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "scripts"))
import memstore as ms  # noqa: E402

_MANAGED_PREFIXES = ("memory_", "user_", "holo_fact_", "jobapp_", "metrics_", "gist_")


def _h(t):
    return hashlib.sha256(t.encode()).hexdigest()[:12]


def _operational_categories():
    """Fact categories excluded from default recall (config plugins.hybrid)."""
    try:
        import yaml
        with open(HERMES_HOME / "config.yaml", encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        cats = ((raw.get("plugins") or {}).get("hybrid") or {}).get("operational_categories")
        if isinstance(cats, list):
            return {str(c) for c in cats}
    except Exception:
        pass
    return {"email", "lead"}


def _sections(path):
    if not path.exists():
        return []
    return [s.strip() for s in path.read_text(encoding="utf-8", errors="replace").split("§") if s.strip()]


def extract_edges_for_new_facts(con, batch=25):
    """Mine graph edges from facts not yet processed (incremental). This is what
    makes the graph auto-grow from EVERYTHING Hermes learns — chat, fact_store,
    session-end auto-extraction — not just ingested email/calls. Append-only and
    idempotent; only NEW facts hit the LLM."""
    rows = con.execute(
        "SELECT fact_id, content FROM facts WHERE fact_id NOT IN "
        "(SELECT fact_id FROM graph_extracted) ORDER BY fact_id LIMIT ?", (batch,)).fetchall()
    todo = [(r[0], r[1]) for r in rows if r[1]]
    if not todo:
        return 0
    try:
        import memory_ingest as mi
        blob = "\n".join(f"- {c}" for _, c in todo)
        ex = mi.extract(blob, "facts", {})
        if ex.get("nodes") or ex.get("edges"):
            mi._append_overlay(ex["nodes"], ex["edges"])  # canonicalize + write to memstore
    except Exception as e:
        print(f"  [warn] edge extraction failed: {e}", file=sys.stderr)
        return 0
    for fid, _ in todo:
        con.execute("INSERT OR IGNORE INTO graph_extracted(fact_id) VALUES (?)", (fid,))
    con.commit()
    return len(todo)


def extract_gists_for_new_facts(con, batch=25):
    """LLM-distill each new fact into a one-line gist ("what this is an
    instance of"), store it in fact_gists, and index it as vid gist_<fact_id>
    (text = the FULL fact content, embedding computed FROM the gist) so recall
    can match on essence rather than surface wording. Incremental via the
    gist_extracted watermark; on LLM failure nothing is watermarked and the
    batch retries next run. Gist VECTORS are only written for non-operational
    facts so the CRM quarantine holds; gist TEXT is stored for every fact."""
    rows = con.execute(
        "SELECT fact_id, content, category FROM facts WHERE fact_id NOT IN "
        "(SELECT fact_id FROM gist_extracted) ORDER BY fact_id LIMIT ?", (batch,)).fetchall()
    todo = [(r[0], r[1], r[2] or "general") for r in rows if r[1]]
    if not todo:
        return 0
    gists = {}
    try:
        import urllib.request
        import memory_ingest as mi
        api_key = mi._read_env_key("OPENROUTER_API_KEY")
        if not api_key:
            print("  [skip] gists: no OPENROUTER_API_KEY", file=sys.stderr)
            return 0
        listing = "\n".join(f'{fid}: {c[:300]}' for fid, c, _ in todo)
        system = (
            "For each numbered fact, output one JSONL line: "
            '{"id": <number>, "gist": "..."} — the gist is ONE short line stating what '
            "the fact is an instance of, abstracted away from its specifics (names, "
            "dates, amounts). Example: 'Jordan's birthday is March 3' -> "
            "'a close person's recurring important date'. Output nothing else."
        )
        payload = json.dumps({
            "model": mi.INGEST_MODEL, "temperature": 0, "max_tokens": 1600,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": listing}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            content = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        for line in content.splitlines():
            line = line.strip().strip(",").lstrip("`").strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if isinstance(o, dict) and isinstance(o.get("gist"), str):
                try:
                    gists[int(o.get("id"))] = o["gist"].strip()[:200]
                except Exception:
                    continue
    except Exception as e:
        print(f"  [warn] gist extraction failed: {e}", file=sys.stderr)
        return 0

    ops = _operational_categories()
    done = 0
    embed_batch = []
    for fid, fact_content, category in todo:
        gist = gists.get(fid) or fact_content[:120]  # deterministic fallback
        con.execute("INSERT OR REPLACE INTO fact_gists(fact_id, gist) VALUES (?,?)",
                    (fid, gist))
        con.execute("INSERT OR IGNORE INTO gist_extracted(fact_id) VALUES (?)", (fid,))
        if category not in ops:
            embed_batch.append((fid, fact_content, gist))
        done += 1
    if embed_batch:
        try:
            vecs = ms.embed([g for _, _, g in embed_batch])
            for (fid, fact_content, _), vec in zip(embed_batch, vecs):
                ms.upsert_vector(con, f"gist_{fid}", fact_content,
                                 "fact_gists", "gist", embedding=vec)
        except Exception as e:
            print(f"  [warn] gist embedding failed: {e}", file=sys.stderr)
    con.commit()
    return done


def main():
    con = ms.connect()
    ms.init_schema(con)
    have = ms.vector_vids(con)
    want = set()
    added = 0

    def _ensure(vid, text, source, vtype):
        nonlocal added
        want.add(vid)
        if vid not in have and text:
            ms.upsert_vector(con, vid, text, source, vtype)
            added += 1

    # MEMORY.md / USER.md sections
    for fname, pref, src, typ in (("MEMORY.md", "memory_", "MEMORY.md", "agent_note"),
                                  ("USER.md", "user_", "USER.md", "user_profile")):
        for sec in _sections(HERMES_HOME / "memories" / fname):
            _ensure(f"{pref}{_h(sec)}", sec, src, typ)

    # Holographic facts
    fc = sqlite3.connect(str(HERMES_HOME / "memory_store.db"))
    fc.row_factory = sqlite3.Row
    try:
        rows = fc.execute("SELECT fact_id, content, category FROM facts").fetchall()
    except Exception:
        rows = []
    fc.close()
    ops = _operational_categories()
    live_fact_ids = set()
    for r in rows:
        text = (r["content"] or "").strip()
        if text:
            live_fact_ids.add(r["fact_id"])
            _ensure(f"holo_fact_{r['fact_id']}", text, "memory_store.db", r["category"] or "general")
            # Gist vectors ride along: keep them for live non-operational facts
            # (creation happens in extract_gists_for_new_facts).
            if (r["category"] or "general") not in ops:
                want.add(f"gist_{r['fact_id']}")

    # Keep vec_items.type in sync with the fact's CURRENT category (also
    # migrates legacy rows written with type='holographic_fact').
    try:
        con.execute(
            "UPDATE vec_items SET type = COALESCE("
            "(SELECT f.category FROM facts f WHERE 'holo_fact_'||f.fact_id = vec_items.vid), type) "
            "WHERE vid LIKE 'holo_fact_%'")
    except Exception as e:
        print(f"  [warn] vec type sync failed: {e}", file=sys.stderr)

    # Drop gist bookkeeping for facts that no longer exist.
    try:
        con.execute("DELETE FROM fact_gists WHERE fact_id NOT IN (SELECT fact_id FROM facts)")
        con.execute("DELETE FROM gist_extracted WHERE fact_id NOT IN (SELECT fact_id FROM facts)")
    except Exception as e:
        print(f"  [warn] gist cleanup failed: {e}", file=sys.stderr)

    # fact_store.json (job applications + metrics)
    fsj = HERMES_HOME / "fact_store.json"
    if fsj.exists():
        try:
            data = json.loads(fsj.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for app in data.get("job applications", []):
            text = app if isinstance(app, str) else json.dumps(app, ensure_ascii=False)
            if text:
                _ensure(f"jobapp_{_h(text)}", text, "fact_store.json", "job_application")
        metrics = data.get("job application metrics")
        if metrics:
            mtext = json.dumps(metrics, ensure_ascii=False)
            _ensure(f"metrics_{_h(mtext)}", f"Job application metrics: {mtext}", "fact_store.json", "job_metrics")

    # Prune vectors whose source is gone (only ids this script owns).
    pruned = 0
    managed = {v for v in have if v.startswith(_MANAGED_PREFIXES)}
    for vid in managed - want:
        ms.delete_vector(con, vid)
        pruned += 1

    con.commit()
    edged = extract_edges_for_new_facts(con)
    gisted = extract_gists_for_new_facts(con)
    st = ms.stats(con)
    print(f"memstore_sync: +{added} vectors, -{pruned} pruned, edges-mined-from {edged} facts, "
          f"gists {gisted} | total {st['vectors']} vectors, {st['edges']} edges")
    con.close()


if __name__ == "__main__":
    main()
