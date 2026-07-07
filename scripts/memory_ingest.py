#!/usr/bin/env python3
"""memory_ingest.py, distill a document (email, call transcript, note) into memory.

Pipeline role: documents IN -> SALIENT facts -> holographic store (memory_store.db).
The existing crons then propagate automatically:
  memstore_sync.py -> sqlite-vec (semantic/vector recall; legacy/memory_import.py retired)
  memstore graph   -> knowledge_graph tables (legacy/build_kg.py retired)

Raw text is NEVER stored as a fact (that is exactly the noise the consolidation
pass strips). Only distilled, durable facts are kept, so the graph and vector
index enrich themselves from your inbox and calls without manual entry.

Idempotent: an ingest ledger (~/.hermes/ingest/ledger.json) records processed
document ids, so re-runs are no-ops.

CLI:
  memory_ingest.py --text "<body>" --source email --id <msgid> --meta from=x --meta subject=y
  memory_ingest.py --spool ~/.hermes/ingest/emails.jsonl
  add --sync to immediately refresh vector index + graph (otherwise the 30m cron does it)
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
HERMES_AGENT = HERMES_HOME / "hermes-agent"
INGEST_DIR = HERMES_HOME / "ingest"
LEDGER = INGEST_DIR / "ledger.json"
KG_DIR = HERMES_HOME / "knowledge_graph"
INGEST_OVERLAY = KG_DIR / "ingest_overlay.json"  # vestigial: graph writes go straight to memstore now
INGEST_MODEL = os.environ.get("INGEST_MODEL", "google/gemini-2.5-flash-lite")

sys.path.insert(0, str(HERMES_AGENT))  # for plugins.memory.holographic.store
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for entity_resolve


def _read_env_key(name: str) -> str:
    v = os.environ.get(name, "")
    if v:
        return v
    envf = HERMES_HOME / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _load_ledger() -> set:
    if LEDGER.exists():
        try:
            return set(json.loads(LEDGER.read_text()))
        except Exception:
            return set()
    return set()


def _save_ledger(ids: set) -> None:
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(sorted(ids)))


def extract(text: str, source: str, meta: dict) -> dict:
    """LLM-distill a document into durable facts + graph entities/edges (JSONL)."""
    empty = {"facts": [], "nodes": [], "edges": []}
    api_key = _read_env_key("OPENROUTER_API_KEY")
    if not api_key:
        print("  [skip] no OPENROUTER_API_KEY", file=sys.stderr)
        return empty
    ctx = ", ".join(f"{k}={v}" for k, v in (meta or {}).items() if v)
    system = (
        "You distill a document into memory. Output JSONL, one compact JSON object per "
        "line, nothing else (no prose, no markdown). Emit three kinds of line:\n"
        '{"fact":"..."}  a durable, self-contained third-person statement worth '
        "remembering; name its subject so it stands alone (e.g. \"Acme Corp's "
        "platform go-live is targeted for the week of July 13\", not \"go-live is week of "
        "July 13\").\n"
        '{"node":{"name":"X","type":"person|company|project|location|job|document|idea"}}  '
        "a concrete named entity mentioned. Job titles/roles are type job, not idea.\n"
        '{"node":{"name":"Trip to Prague (2026-08-13)","type":"event","attrs":{"start":"2026-08-13","end":"2026-08-22","location":"Prague"}}}  '
        "a dated occurrence (trip, party, meeting, flight, deadline). Include the start "
        "date in the name so recurring/similar occurrences stay distinct; put ISO dates "
        "in attrs (end optional). Emit edges tying it together: participants "
        "<person> ATTENDS <event>, place <event> LOCATED_IN <location>.\n"
        '{"edge":{"source":"X","target":"Y","relation":"..."}}  a relationship between '
        "two named entities. relation MUST be one of exactly these:\n"
        "LOCATED_IN, LIVES_IN, TRAVELS_TO, WORKS_FOR, LEADS, FOUNDED, OWNS, PART_OF, "
        "MEMBER_OF, AFFILIATED_WITH, APPLIED_TO, HIRING, POSTED_ON, CONTACT_AT, "
        "CLIENT_OF, VENDOR_OF, PARTNER_OF, PROSPECT, CONTACTED, OWES, OFFERS, "
        "REFERRAL_FROM, IN_INDUSTRY, TARGETS, HAS_PROJECT, USES, SPOUSE_OF, PARENT_OF, "
        "ATTENDS, ABOUT, INVOLVES, INTERESTED_IN, RELATED_TO.\n"
        "The subject acts: <person> WORKS_FOR <company>, <company> HIRING <job>, "
        "<person> APPLIED_TO <job>, <company> OFFERS <service>, <company> LOCATED_IN "
        "<place>, <job> POSTED_ON <platform>. NEVER emit passive/inverse forms "
        "(OFFERED_BY, POSTED_BY, LED_BY, FOUNDED_BY, ROLE_AT), swap source/target "
        "instead. If no relation fits, use RELATED_TO. Salary, dates, and deadlines are "
        "node attrs, not edges.\n"
        "SKIP greetings, signatures, marketing, legal boilerplate, tracking numbers, and "
        "ephemera. If nothing is worth remembering, output nothing. At most 8 facts, "
        "12 nodes, 12 edges."
    )
    user = f"SOURCE: {source}\nCONTEXT: {ctx}\n\nDOCUMENT:\n{text[:6000]}"
    payload = json.dumps({
        "model": INGEST_MODEL,
        "temperature": 0,
        "max_tokens": 1600,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body = json.loads(r.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [warn] extraction failed: {e}", file=sys.stderr)
        return empty

    facts, nodes, edges = [], [], []
    for line in content.splitlines():
        line = line.strip().strip(",").lstrip("`").strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        if isinstance(o.get("fact"), str):
            f = o["fact"].strip()
            if 8 <= len(f) <= 400:
                facts.append(f)
        elif isinstance(o.get("node"), dict):
            nodes.append(o["node"])
        elif isinstance(o.get("edge"), dict):
            edges.append(o["edge"])
    return {"facts": facts, "nodes": nodes, "edges": edges}


def _append_overlay(nodes: list, edges: list, source_ref: str = "") -> None:
    """Write canonicalized nodes/edges straight into the unified store (memstore).
    Same-entity variants collapse to one node via entity_resolve; the graph is
    maintained incrementally, no overlay JSON, no periodic rebuild."""
    try:
        import memstore as _ms
        from entity_resolve import Resolver, canon_relation as _cr
        con = _ms.connect()
        _ms.init_schema(con)
        rz = Resolver()
    except Exception as e:
        print(f"  [warn] graph write skipped: {e}", file=sys.stderr)
        return

    def _canon(name, htype=None):
        nm = (name or "").strip()
        if not nm:
            return ""
        try:
            return rz.canonical(nm, hint_type=htype)
        except Exception:
            return nm

    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        ntype = n.get("type", "concept")
        ht = ntype if ntype in ("person", "company", "project", "location",
                                "job", "document", "idea", "event") else None
        nm = _canon(n.get("name"), ht)
        if not nm:
            continue
        attrs = {}
        if isinstance(n.get("attrs"), dict):
            attrs = {k: v for k, v in n["attrs"].items() if isinstance(v, (str, int, float, bool))}
        _ms.add_node(con, nm, ntype, attrs)

    for e in edges or []:
        if not isinstance(e, dict):
            continue
        s = _canon(e.get("source"))
        t = _canon(e.get("target"))
        raw_rel = str(e.get("relation") or "").strip()
        rel, flipped = _cr(raw_rel)
        if flipped:
            s, t = t, s
        # Keep the extractor's original wording only when the canonizer had
        # to change it, a non-empty rel_orig is the drift signal.
        changed = flipped or rel != raw_rel.upper().replace(" ", "_")
        if s and t and s != t:
            _ms.add_edge(con, s, rel, t, "ingest", source_ref=source_ref,
                         rel_orig=raw_rel[:60] if changed else "")

    # Close-call fuzzy matches (blend evidence): entities that ALMOST merged
    # get a reviewable POSSIBLE_ALIAS edge instead of a silent decision.
    try:
        for a, b, score in getattr(rz, "pending_aliases", []):
            if a and b and a != b:
                _ms.add_edge(con, a, "POSSIBLE_ALIAS", b, "alias-candidate")
    except Exception as e:
        print(f"  [warn] alias-candidate write failed: {e}", file=sys.stderr)

    try:
        rz.save()
    except Exception:
        pass
    con.commit()
    con.close()


def _set_fact_provenance(fact_ids: list, source_ref: str) -> None:
    """Stamp source_ref on facts without touching upstream store.py.
    The empty-guard keeps first-assertion semantics: a duplicate fact
    returned by add_fact keeps the provenance of the document that first
    asserted it."""
    ids = [int(i) for i in fact_ids if i]
    if not ids or not source_ref:
        return
    try:
        import sqlite3
        con = sqlite3.connect(str(HERMES_HOME / "memory_store.db"), timeout=30)
        con.execute("PRAGMA busy_timeout=30000")
        con.executemany(
            "UPDATE facts SET source_ref=? WHERE fact_id=? "
            "AND (source_ref IS NULL OR source_ref='')",
            [(source_ref, i) for i in ids])
        con.commit()
        con.close()
    except Exception as e:
        print(f"  [warn] provenance stamp failed: {e}", file=sys.stderr)


_STORE = None


def _store():
    global _STORE
    if _STORE is None:
        from plugins.memory.holographic.store import MemoryStore
        _STORE = MemoryStore(db_path=str(HERMES_HOME / "memory_store.db"))
    return _STORE


def ingest_document(doc: dict, ledger: set) -> dict:
    """doc = {id, source, text, meta}. Returns {facts, nodes, edges} ({} if skipped)."""
    did = str(doc.get("id") or "")
    if not did or did in ledger:
        return {}
    text = (doc.get("text") or "").strip()
    source = doc.get("source", "note")
    meta = doc.get("meta", {}) or {}
    if len(text) < 20:
        ledger.add(did)
        return {}

    source_ref = f"{source}:{did}"[:200]
    ex = extract(text, source, meta)
    facts, nodes, edges = ex["facts"], ex["nodes"], ex["edges"]
    if facts:
        st = _store()
        tagbits = [source] + [f"{k}:{v}" for k, v in meta.items()
                              if k in ("from", "sender", "number", "direction") and v]
        tags = ",".join(tagbits)[:120]
        new_ids = []
        for f in facts:
            try:
                new_ids.append(st.add_fact(f, category=source, tags=tags))
            except Exception as e:
                print(f"  [warn] add_fact failed: {e}", file=sys.stderr)
        _set_fact_provenance(new_ids, source_ref)
    if nodes or edges:
        _append_overlay(nodes, edges, source_ref=source_ref)
    ledger.add(did)
    return ex


def _sync() -> None:
    import subprocess
    py = str(HERMES_AGENT / "venv" / "bin" / "python3")
    env = {**os.environ, "HERMES_HOME": str(HERMES_HOME)}
    subprocess.run([py, str(HERMES_HOME / "scripts" / "memstore_sync.py")], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--source", default="note")
    ap.add_argument("--id")
    ap.add_argument("--spool")
    ap.add_argument("--meta", action="append", default=[], help="k=v (repeatable)")
    ap.add_argument("--sync", action="store_true", help="refresh vector+graph now")
    a = ap.parse_args()

    ledger = _load_ledger()
    docs = []
    if a.spool:
        p = Path(a.spool)
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except Exception:
                        pass
    elif a.text:
        meta = dict(kv.split("=", 1) for kv in a.meta if "=" in kv)
        did = a.id or hashlib.sha256(a.text.encode()).hexdigest()[:16]
        docs.append({"id": did, "source": a.source, "text": a.text, "meta": meta})
    else:
        print("need --text or --spool")
        return

    ndocs = nf = 0
    for d in docs:
        ex = ingest_document(d, ledger)
        facts = ex.get("facts", []) if ex else []
        if ex and (facts or ex.get("nodes") or ex.get("edges")):
            ndocs += 1
            nf += len(facts)
            print(f"  {d.get('source')} {d.get('id')}: {len(facts)} facts, "
                  f"{len(ex.get('nodes', []))} nodes, {len(ex.get('edges', []))} edges")
            for f in facts:
                print(f"     - {f[:88]}")
    _save_ledger(ledger)
    print(f"done: {ndocs} docs enriched, {nf} facts added")
    if a.sync and nf:
        print("syncing vector index + graph...")
        _sync()


if __name__ == "__main__":
    main()
