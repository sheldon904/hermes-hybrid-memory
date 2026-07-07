"""hybrid: holographic facts + semantic vector recall + knowledge-graph memory.

A MemoryProvider that SUBCLASSES the bundled holographic provider, so it keeps
every holographic behavior (fact_store/fact_feedback tools, entity resolution,
trust scoring, auto-extraction at session end) and ADDS on top:

  - recall: vector (semantic) search over all memory (sqlite-vec)
  - graph_query and friends: relationship tools over the knowledge graph
  - prefetch(): auto-injects FTS + vector + graph context every turn,
    char-capped, with retrieval counting and recall logging
  - analogy slot: one structure-similar, surface-different memory per turn
    (HRR vs MiniLM disagreement), Hofstadter-style
  - situation match: situation/playbook nodes matched against the rolling
    conversation window
  - chunk awareness: chunk facts suppress their members in prefetch;
    chunk_expand unpacks them
  - analogize tool: relational-fingerprint analogy search over the graph
  - on_memory_write(): instant vector upsert when the memory tool writes

Config in $HERMES_HOME/config.yaml:
    plugins:
      hybrid:
        prefetch_char_cap: 2000       # hard cap on the injected block (0 = off)
        count_retrievals: true        # bump facts.retrieval_count on surfacing
        recall_log_enabled: true      # write recall_log rows for the nightly pass
        operational_categories: [email, lead]   # excluded from default prefetch
        operational_vec_types: [email, lead, job_application]
        rolling_window: 3             # user messages in the situation window
        analogy_slot: true
        analogy_hrr_min: 0.62         # normalized HRR sim floor
        analogy_surface_max: 0.30     # MiniLM cosine ceiling (surface-different)
        analogy_min_query_chars: 40
        situation_sim_min: 0.45
        analogize_enabled: true

Everything degrades gracefully: if the vector store, graph, numpy/HRR, or the
holographic base is unavailable, the provider falls back to the remaining
layers (worst case: exactly the bundled holographic behavior).

Lives in $HERMES_HOME/plugins/ (update-safe). Activate with:
    memory:
      provider: hybrid
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Subclass holographic when available; fall back to the bare ABC so the agent
# still gets vector+graph memory even if the bundled provider ever moves.
try:
    from plugins.memory.holographic import HolographicMemoryProvider as _Base
    _HOLO = True
except Exception as e:  # pragma: no cover
    from agent.memory_provider import MemoryProvider as _Base
    _HOLO = False
    logger.warning("hybrid: holographic base unavailable (%s); vector+graph only", e)

# HRR math (phase vectors) from the bundled plugin, used for the analogy slot.
try:
    from plugins.memory.holographic import holographic as _hrrmod
    import numpy as _np
    _HRR_OK = True
except Exception as e:  # pragma: no cover
    _hrrmod = None
    _np = None
    _HRR_OK = False
    logger.debug("hybrid: HRR module unavailable (%s); analogy slot disabled", e)

try:
    from tools.registry import tool_error
except Exception:  # pragma: no cover
    def tool_error(msg: str) -> str:
        return json.dumps({"error": msg})

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
ENTITY_INDEX_PATH = HERMES_HOME / "knowledge_graph" / "entity_index.json"

# Unified store: sqlite-vec vectors + graph edges, all inside memory_store.db.
import sys as _sys
_sys.path.insert(0, str(HERMES_HOME / "scripts"))
try:
    import memstore as _ms
except Exception as _e:  # pragma: no cover
    _ms = None
    logger.warning("hybrid: memstore unavailable (%s); semantic/graph degraded", _e)

try:
    from entity_resolve import normalize as _er_normalize
except Exception:  # pragma: no cover
    def _er_normalize(name):
        return " ".join(str(name or "").lower().split())

# Auto-injection only surfaces vector hits at/above this cosine similarity, so
# weak matches never pollute the prompt. The explicit recall tool is looser.
PREFETCH_SIM_THRESHOLD = 0.40
RECALL_SIM_FLOOR = 0.25

# Allow disabling the in-process embedder (RAM) without losing fact/graph memory.
_VECTOR_DISABLED = os.environ.get("HYBRID_VECTOR", "1") == "0"

REASON_MODEL = os.environ.get("HYBRID_REASON_MODEL", "google/gemini-2.5-flash-lite")

_DEFAULT_HCFG = {
    "prefetch_char_cap": 2000,
    "count_retrievals": True,
    "recall_log_enabled": True,
    "operational_categories": ["email", "lead"],
    "operational_vec_types": ["email", "lead", "job_application"],
    "rolling_window": 3,
    "analogy_slot": True,
    # HRR phase-similarity over bundled bags-of-words compresses into a tight
    # band around 0.5 (superposition noise), so the gate is RELATIVE: a
    # candidate must stand analogy_z_min standard deviations above the mean
    # similarity across all facts, plus clear the absolute sanity floor.
    "analogy_hrr_min": 0.50,
    "analogy_z_min": 2.0,
    "analogy_surface_max": 0.30,
    "analogy_min_query_chars": 40,
    "situation_sim_min": 0.42,
    "analogize_enabled": True,
}


def _load_hybrid_config() -> dict:
    cfg = dict(_DEFAULT_HCFG)
    path = HERMES_HOME / "config.yaml"
    try:
        import yaml
        with open(path, encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        user = ((raw.get("plugins") or {}).get("hybrid") or {})
        if isinstance(user, dict):
            cfg.update(user)
    except Exception as e:
        logger.debug("hybrid: config load failed (%s); using defaults", e)
    return cfg


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())[:160]


def _tokens(s: str) -> set:
    return {t.strip(".,!?;:\"'()[]{}").lower() for t in (s or "").split() if len(t) > 2}


_STOPWORDS = frozenset(
    "the a an and or but if then else when while for to of in on at by with from as is "
    "are was were be been being am do does did doing have has had having i you he she "
    "it we they me him her us them my your his its our their this that these those "
    "there here what which who whom whose why how not no nor so too very can will "
    "just should would could about into over under again further once same than "
    "any all both each few more most other some such only own out up down off".split())


def _content_words(s: str) -> str:
    """Strip stopwords/short tokens so the HRR probe keys on distinctive words."""
    toks = [t.strip(".,!?;:\"'()[]{}") for t in (s or "").lower().split()]
    return " ".join(t for t in toks if len(t) > 2 and t not in _STOPWORDS)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Entity-candidate hygiene. The bundled holographic extractor is regex-only
# (capitalized multi-word phrases + quoted spans) and produces junk like
# "With Alex", "Arrive Prague" and apostrophe-spanning blobs ("Alex's
# ... Kim's" captured as one 'quoted' span). These filters wrap it.
_ENTITY_MAX_LEN = 60
_ENTITY_LEAD_JUNK = frozenset(
    "with for from and the arrive arriving arrives depart departs departing "
    "leave leaves leaving visit visits visiting return returns returning".split())


def _clean_entity_candidates(names) -> List[str]:
    """Filter raw regex entity candidates: drop newline/overlong spans and
    lowercase multi-word blobs; strip junk leading verbs/prepositions
    ('With Mac' -> 'Mac'). Preserves first-seen order, dedupes case-insensitively."""
    out: List[str] = []
    seen: Set[str] = set()

    def _add(n: str) -> None:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)

    for raw in names or []:
        raw_s = str(raw)
        if "\n" in raw_s:
            continue
        n = " ".join(raw_s.split()).strip()
        if not n or len(n) > _ENTITY_MAX_LEN:
            continue
        words = n.split()
        # Apostrophe-rule blobs surface as lowercase many-word spans.
        if n[0].islower() and len(words) > 3:
            continue
        first = words[0].strip(".,!?;:\"'()").lower()
        if first in _ENTITY_LEAD_JUNK and len(words) > 1:
            rest = " ".join(words[1:])
            if rest[:1].isupper():
                _add(rest)
            continue
        _add(n)
    return out


def _read_env_key(name: str) -> str:
    v = os.environ.get(name, "")
    if v:
        return v
    envf = HERMES_HOME / ".env"
    if envf.exists():
        try:
            for line in envf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def _cap_blocks(blocks: List[str], cap: int) -> Tuple[str, Set[str]]:
    """Assemble blocks in priority order, keeping whole lines until `cap` chars.

    Returns (text, set of included non-header lines). cap <= 0 disables capping.
    """
    included: Set[str] = set()
    out_parts: List[str] = []
    used = 0
    full = False
    for block in blocks:
        if not block or full:
            continue
        kept: List[str] = []
        for ln in block.splitlines():
            cost = len(ln) + 1
            if cap > 0 and used + cost > cap:
                full = True
                break
            kept.append(ln)
            used += cost
            if not ln.startswith("#"):
                included.add(ln)
        # A block that kept only its header contributes nothing.
        if kept and any(not l.startswith("#") and l.strip() for l in kept):
            out_parts.append("\n".join(kept))
        else:
            for ln in kept:
                included.discard(ln)
    return "\n\n".join(out_parts).strip(), included


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "recall",
    "description": (
        "Semantic (vector) search over EVERYTHING Hermes remembers, agent notes, "
        "the user profile, stored facts, job applications. Use this for fuzzy or "
        "conceptual recall when you don't have exact keywords (it finds by meaning). "
        "Complements fact_store (keyword/entity) and graph_query (relationships). "
        "Relevant memory is already auto-injected each turn; call recall to dig deeper. "
        "Operational noise (email/lead exhaust) is included by default here even though "
        "it is excluded from auto-injection; set include_operational=false to skip it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of what to recall."},
            "k": {"type": "integer", "description": "Max results (default 5, max 15)."},
            "include_operational": {"type": "boolean",
                                    "description": "Include email/lead exhaust (default true)."},
        },
        "required": ["query"],
    },
}

GRAPH_QUERY_SCHEMA = {
    "name": "graph_query",
    "description": (
        "Query the memory knowledge graph for how an entity connects to others "
        "(people, companies, projects, places). Returns the entity's neighbors and "
        "the relationship on each edge. Use for relational questions like "
        "'who/what is connected to X', 'X's clients', 'X's projects'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name, e.g. 'Acme Labs' or 'Alex'."},
            "depth": {"type": "integer", "description": "Hops to traverse (default 1, max 2)."},
        },
        "required": ["entity"],
    },
}

GRAPH_PATH_SCHEMA = {
    "name": "graph_path",
    "description": (
        "Find how two entities are connected in the memory knowledge graph, the shortest "
        "chain of relationships between them (multi-hop). Use for 'how is X connected to Y', "
        "'what links A and B', warm-intro / connection questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Start entity."},
            "target": {"type": "string", "description": "End entity."},
            "max_hops": {"type": "integer", "description": "Max chain length to report (default 6)."},
        },
        "required": ["source", "target"],
    },
}

GRAPH_CONNECTIONS_SCHEMA = {
    "name": "graph_connections",
    "description": (
        "Given several entities, show how they relate across the graph, pairwise connection "
        "paths and any shared neighbors (common city, industry, contact, project). Use for "
        "'how do these relate', 'what do these prospects have in common', clustering questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"},
                         "description": "Two or more entity names."},
        },
        "required": ["entities"],
    },
}

GRAPH_REASON_SCHEMA = {
    "name": "graph_reason",
    "description": (
        "Reason over a focused region of the knowledge graph to answer a relational / multi-hop "
        "question (e.g. 'which prospects look most like our best client', 'how is this company "
        "positioned relative to X'). Pulls the relevant subgraph and analyses it. Heavier than "
        "graph_query, use when the answer needs inference across multiple relationships."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The relational question to reason about."},
            "entities": {"type": "array", "items": {"type": "string"},
                         "description": "Entities to centre the reasoning on (recommended)."},
        },
        "required": ["question"],
    },
}

CHUNK_EXPAND_SCHEMA = {
    "name": "chunk_expand",
    "description": (
        "Unpack a chunk memory (category='chunk') into the member facts it summarizes. "
        "Chunks appear in auto-injected memory as single compressed lines; call this to "
        "see the underlying detail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_fact_id": {"type": "integer", "description": "fact_id of the chunk to expand."},
        },
        "required": ["chunk_fact_id"],
    },
}

SITUATION_STORE_SCHEMA = {
    "name": "situation_store",
    "description": (
        "Manage situation-pattern memories, named recurring situations with a playbook "
        "(e.g. 'vendor ghosting after verbal yes'). When a conversation resembles a stored "
        "situation, its playbook is auto-injected. Use 'add' when you and the user recognize "
        "a recurring pattern worth naming."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "remove"]},
            "name": {"type": "string", "description": "Short situation label (required for add/remove)."},
            "description": {"type": "string",
                            "description": "What the situation looks like when it occurs (required for add; used for matching)."},
            "playbook": {"type": "string", "description": "What to do / remember when it recurs (required for add)."},
        },
        "required": ["action"],
    },
}

DECISION_LOG_SCHEMA = {
    "name": "decision_log",
    "description": (
        "Record decision traces: what you recommended vs what the user chose. "
        "Call action='record' when you present the user a RECOMMENDATION, PROPOSAL, "
        "or set of options on something that matters (a purchase, a plan, an approach, "
        "an outreach draft), not for ordinary answers. When the user later accepts, "
        "rejects, or modifies it, call action='resolve' with the outcome and their "
        "reason if they gave one. These traces improve future recommendations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["record", "resolve", "list"]},
            "kind": {"type": "string", "enum": ["recommendation", "option-set", "proposal"],
                     "description": "record: what shape of decision this is."},
            "proposal": {"type": "string",
                         "description": "record: the recommendation, in one line (required)."},
            "options_shown": {"type": "array", "items": {"type": "string"},
                              "description": "record: alternatives presented, if any."},
            "decision_id": {"type": "integer",
                            "description": "resolve: id from record (omit to resolve the latest pending in this session)."},
            "outcome": {"type": "string", "enum": ["accepted", "rejected", "modified"],
                        "description": "resolve: what the user did (required)."},
            "chosen": {"type": "string",
                       "description": "resolve: what the user actually went with (esp. when modified)."},
            "reason": {"type": "string", "description": "resolve: the user's stated reason, if any."},
        },
        "required": ["action"],
    },
}

ONTOLOGY_REVIEW_SCHEMA = {
    "name": "ontology_review",
    "description": (
        "Review machine-proposed knowledge-graph changes awaiting the user's decision: "
        "entity alias merges ('are X and Y the same entity?') and proposed categories "
        "(INSTANCE_OF groupings). Use when the user replies to the weekly ontology nudge "
        "or asks about memory/graph proposals. action='list' shows the pending queue. "
        "Approve an alias ONLY when confident both names are the same real-world "
        "entity, when unsure, ask the user. Approvals permanently merge graph nodes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "approve", "reject"]},
            "kind": {"type": "string", "enum": ["alias", "category"],
                     "description": "approve/reject: which queue."},
            "a": {"type": "string",
                  "description": "alias: the name to merge AWAY (loser/variant)."},
            "b": {"type": "string",
                  "description": "alias: the name to KEEP (winner/canonical)."},
            "label": {"type": "string", "description": "category: the category label."},
        },
        "required": ["action"],
    },
}

ANALOGIZE_SCHEMA = {
    "name": "analogize",
    "description": (
        "Find entities whose RELATIONAL STRUCTURE in the knowledge graph resembles a given "
        "entity (or a described situation), while being surface-different, Hofstadter-style "
        "analogy search. Use for 'which prospects are shaped like our best client', "
        "'what past situation does this resemble'. Set explain=true for an LLM mapping "
        "explanation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Source entity to analogize from."},
            "text": {"type": "string", "description": "Free-text situation (alternative to entity)."},
            "k": {"type": "integer", "description": "Max mappings (default 5)."},
            "explain": {"type": "boolean", "description": "Add an LLM explanation of the best mapping."},
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class HybridMemoryProvider(_Base):
    """Holographic + semantic vector + knowledge-graph memory."""

    def __init__(self, config: dict | None = None):
        if _HOLO:
            super().__init__(config)
        self._hcfg = _load_hybrid_config()
        self._memcon = None
        self._vector_ok = False
        self._graph = None
        self._alias_map: Dict[str, str] = {}
        self._rolling: collections.deque = collections.deque(
            maxlen=max(1, int(self._hcfg.get("rolling_window", 3) or 1)))
        self._turn_number = 0
        self._hrr_cache = None  # (key, ids ndarray, matrix ndarray, {fact_id: content})

    @property
    def name(self) -> str:
        return "hybrid"

    def is_available(self) -> bool:
        return True  # holographic/SQLite always works; vector+graph are best-effort

    # -- config helpers --------------------------------------------------------

    def _op_categories(self) -> Set[str]:
        return {str(c) for c in (self._hcfg.get("operational_categories") or [])}

    def _op_vec_types(self) -> Set[str]:
        return {str(c) for c in (self._hcfg.get("operational_vec_types") or [])}

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        if _HOLO:
            super().initialize(session_id, **kwargs)
        self._session_id = session_id
        agent_context = kwargs.get("agent_context", "primary")
        self._rolling.clear()
        self._turn_number = 0
        self._hrr_cache = None

        # Unified store: sqlite-vec vectors + graph edges in memory_store.db
        if not _VECTOR_DISABLED and _ms is not None:
            try:
                self._memcon = _ms.connect()
                _ms.init_schema(self._memcon)
                # The turn path should give up fast on a locked DB (drop the
                # log, never stall the turn); background jobs keep the long
                # timeout on their own connections.
                self._memcon.execute("PRAGMA busy_timeout=2000")
                self._vector_ok = True
            except Exception as e:
                logger.warning("hybrid: unified store unavailable (%s)", e)
                self._vector_ok = False
            # Warm the embedder for interactive sessions so turn 1 isn't slow.
            if self._vector_ok and agent_context == "primary":
                try:
                    _ms._ef()
                except Exception as e:
                    logger.debug("hybrid: embedder warm failed (%s)", e)

        self._load_graph()
        try:
            self._install_entity_filter()
        except Exception as e:
            logger.debug("hybrid: entity filter install failed (%s)", e)

    def _install_entity_filter(self) -> None:
        """Wrap the holographic store's regex `_extract_entities` with
        `_clean_entity_candidates`, and rescue known single-word proper nouns
        (Prague, Chicago) the multi-word regex misses via the alias index.
        Wrapping the store INSTANCE keeps this update-safe: if upstream
        renames the method this silently no-ops and behavior degrades to
        stock extraction."""
        store = getattr(self, "_store", None)
        orig = getattr(store, "_extract_entities", None)
        if store is None or not callable(orig) or getattr(store, "_hybrid_entity_filter", False):
            return
        prov = self

        def _filtered(text: str) -> list:
            cleaned = _clean_entity_candidates(orig(text))
            try:
                have = {c.lower() for c in cleaned}
                for tok in {t.strip(".,!?;:\"'()") for t in str(text or "").split()}:
                    if len(tok) >= 4 and tok[:1].isupper():
                        canon = prov._alias_map.get(_er_normalize(tok))
                        if canon and canon.lower() not in have:
                            cleaned.append(canon)
                            have.add(canon.lower())
            except Exception:
                pass
            return cleaned

        store._extract_entities = _filtered
        store._hybrid_entity_filter = True

    def _load_graph(self) -> None:
        self._graph = None
        self._alias_map = {}
        try:
            if self._memcon is not None and _ms is not None:
                self._graph = _ms.load_graph(self._memcon)
        except Exception as e:
            logger.debug("hybrid: graph load failed (%s)", e)
            self._graph = None
        # Alias index (write-time canonicalization) reused at read time so
        # 'BWL' or 'Mac' in a query still lights up the canonical node.
        try:
            if ENTITY_INDEX_PATH.exists():
                idx = json.loads(ENTITY_INDEX_PATH.read_text(encoding="utf-8"))
                for key, canon in (idx.get("keys") or {}).items():
                    if not key.startswith("norm:"):
                        continue
                    alias = key[5:].strip()
                    if len(alias) >= 4 and canon:
                        self._alias_map[alias] = canon
        except Exception as e:
            logger.debug("hybrid: alias index load failed (%s)", e)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        try:
            if _HOLO:
                super().on_turn_start(turn_number, message, **kwargs)
        except Exception:
            pass
        self._turn_number = int(turn_number or 0)
        self._push_rolling(message)

    def _push_rolling(self, message: str) -> None:
        if int(self._hcfg.get("rolling_window", 3) or 0) <= 0:
            return
        if not isinstance(message, str):
            return
        snippet = message.strip()[:300]
        if snippet and (not self._rolling or self._rolling[-1] != snippet):
            self._rolling.append(snippet)

    # -- vector helpers ------------------------------------------------------

    def _vector_search(self, query: str, k: int = 5, *,
                       exclude_types: Optional[Set[str]] = None,
                       embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        if not self._vector_ok or self._memcon is None or _ms is None:
            return []
        if not query and embedding is None:
            return []
        try:
            # Over-fetch so we can drop duplicate text and excluded types and
            # still return k distinct hits.
            raw = _ms.query_vectors(self._memcon, query_text=query,
                                    embedding=embedding, k=min(k + 12, 40))
            out, seen = [], set()
            for item in raw:
                if exclude_types and (item.get("type") or "") in exclude_types:
                    continue
                key = _norm(item["text"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({"text": item["text"], "source": item.get("source", "?"),
                            "type": item.get("type", ""), "vid": item.get("vid", ""),
                            "similarity": item["similarity"]})
                if len(out) >= k:
                    break
            return out
        except Exception as e:
            logger.debug("hybrid: vector search failed (%s)", e)
            return []

    @staticmethod
    def _fact_id_from_vid(vid: str) -> Optional[int]:
        for pref in ("holo_fact_", "gist_"):
            if vid and vid.startswith(pref):
                try:
                    return int(vid[len(pref):])
                except Exception:
                    return None
        return None

    # -- graph helpers -------------------------------------------------------

    def _resolve_node(self, entity: str):
        if not self._graph:
            return None
        if entity in self._graph:
            return entity
        el = str(entity).lower()
        for n in self._graph.nodes():
            if str(n).lower() == el:
                return n
        canon = self._alias_map.get(_er_normalize(entity))
        if canon and canon in self._graph:
            return canon
        return None

    def _neighbors(self, entity: str, depth: int = 1) -> List[Dict[str, Any]]:
        node = self._resolve_node(entity)
        if node is None:
            return []
        out, visited, frontier = [], {node}, [node]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for nb in list(self._graph.successors(cur)):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    rel = self._graph.edges[cur, nb].get("relation", "RELATED_TO")
                    out.append({"from": cur, "to": nb, "relation": rel})
                    nxt.append(nb)
                for nb in list(self._graph.predecessors(cur)):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    rel = self._graph.edges[nb, cur].get("relation", "RELATED_TO")
                    out.append({"from": nb, "to": cur, "relation": rel})
                    nxt.append(nb)
            frontier = nxt
        return out

    def _graph_context(self, query: str, max_entities: int = 2) -> str:
        if not self._graph or not query:
            return ""
        ql = query.lower()
        hits, seen = [], set()

        def _add(node):
            if node in seen:
                return
            seen.add(node)
            hits.append(node)

        for node in self._graph.nodes():
            nl = str(node).lower()
            if len(nl) < 4:
                continue  # skip short names to avoid false substring hits
            if nl in ql:
                _add(node)
            if len(hits) >= max_entities:
                break
        if len(hits) < max_entities:
            for alias, canon in self._alias_map.items():
                if len(hits) >= max_entities:
                    break
                if alias in ql and canon in self._graph:
                    _add(canon)
        lines = []
        for ent in hits:
            for c in self._neighbors(ent, depth=1)[:6]:
                lines.append(f"- {c['from']} {c['relation']} {c['to']}")
        if not lines:
            return ""
        return "## Related (knowledge graph)\n" + "\n".join(lines[:10])

    # -- counting + logging ----------------------------------------------------

    def _bump_retrieval(self, fact_ids) -> None:
        if not self._hcfg.get("count_retrievals", True):
            return
        ids = sorted({int(f) for f in fact_ids if f is not None})
        if not ids:
            return
        store = getattr(self, "_store", None)
        conn = getattr(store, "_conn", None)
        lock = getattr(store, "_lock", None)
        if conn is None:
            return
        try:
            qmarks = ",".join("?" * len(ids))
            if lock is not None:
                with lock:
                    conn.execute(
                        f"UPDATE facts SET retrieval_count = retrieval_count + 1 "
                        f"WHERE fact_id IN ({qmarks})", ids)
                    conn.commit()
            else:
                conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 "
                    f"WHERE fact_id IN ({qmarks})", ids)
                conn.commit()
        except Exception as e:
            logger.debug("hybrid: retrieval bump failed (%s)", e)

    def _log_recall(self, query: str, entries: List[Dict[str, Any]]) -> None:
        if not self._hcfg.get("recall_log_enabled", True):
            return
        if not entries or self._memcon is None or _ms is None:
            return
        try:
            sid = getattr(self, "_session_id", "") or ""
            rows = [(sid, self._turn_number, (query or "")[:400], e.get("block", ""),
                     e.get("fact_id"), e.get("vid"), e.get("score")) for e in entries]
            _ms.log_recall(self._memcon, rows)
        except Exception as e:
            logger.debug("hybrid: recall log failed (%s)", e)

    # -- HRR analogy slot ------------------------------------------------------

    def _hrr_matrix(self):
        """Cached (ids, matrix, contents) over non-operational facts with HRR vectors."""
        if not _HRR_OK:
            return None
        store = getattr(self, "_store", None)
        conn = getattr(store, "_conn", None)
        lock = getattr(store, "_lock", None)
        if conn is None:
            return None
        try:
            if lock is not None:
                with lock:
                    key = conn.execute(
                        "SELECT COUNT(*), MAX(fact_id), MAX(updated_at) FROM facts "
                        "WHERE hrr_vector IS NOT NULL").fetchone()
            else:
                key = conn.execute(
                    "SELECT COUNT(*), MAX(fact_id), MAX(updated_at) FROM facts "
                    "WHERE hrr_vector IS NOT NULL").fetchone()
            key = tuple(key)
            if self._hrr_cache is not None and self._hrr_cache[0] == key:
                return self._hrr_cache[1:]
            ops = self._op_categories()
            if lock is not None:
                with lock:
                    rows = conn.execute(
                        "SELECT fact_id, content, category, hrr_vector FROM facts "
                        "WHERE hrr_vector IS NOT NULL").fetchall()
            else:
                rows = conn.execute(
                    "SELECT fact_id, content, category, hrr_vector FROM facts "
                    "WHERE hrr_vector IS NOT NULL").fetchall()
            ids, vecs, contents = [], [], {}
            for r in rows:
                fid, content, category, blob = r[0], r[1], r[2], r[3]
                if (category or "") in ops or not blob:
                    continue
                try:
                    v = _hrrmod.bytes_to_phases(blob)
                except Exception:
                    continue
                ids.append(fid)
                vecs.append(v)
                contents[fid] = content or ""
            if not ids:
                self._hrr_cache = (key, None, None, {})
                return None
            # Keep only the dominant dimension so vstack never fails on a
            # stray vector written with a different hrr_dim.
            dim = collections.Counter(len(v) for v in vecs).most_common(1)[0][0]
            pairs = [(fid, v) for fid, v in zip(ids, vecs) if len(v) == dim]
            if not pairs:
                self._hrr_cache = (key, None, None, {})
                return None
            ids_arr = _np.array([fid for fid, _ in pairs])
            mat = _np.vstack([v for _, v in pairs])
            self._hrr_cache = (key, ids_arr, mat, contents)
            return self._hrr_cache[1:]
        except Exception as e:
            logger.debug("hybrid: HRR matrix build failed (%s)", e)
            return None

    def _recent_analogy_ids(self, limit: int = 20) -> Set[int]:
        if self._memcon is None:
            return set()
        try:
            rows = self._memcon.execute(
                "SELECT fact_id FROM recall_log WHERE block='analogy' AND fact_id IS NOT NULL "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return {int(r[0]) for r in rows}
        except Exception:
            return set()

    def _surface_similarity(self, fact_id: int, content: str,
                            q_emb: Optional[List[float]], q_toks: set) -> float:
        """MiniLM cosine between the query and the fact's stored embedding;
        token-Jaccard fallback when the vector row is missing."""
        if q_emb is not None and self._memcon is not None and _ms is not None:
            try:
                row = self._memcon.execute(
                    "SELECT vec_distance_cosine(embedding, ?) FROM vec_index "
                    "WHERE item_id = (SELECT id FROM vec_items WHERE vid = ?)",
                    (_ms._pack(q_emb), f"holo_fact_{fact_id}")).fetchone()
                if row and row[0] is not None:
                    return 1.0 - float(row[0])
            except Exception:
                pass
        return _jaccard(q_toks, _tokens(content))

    def _analogy_candidate(self, rolling_query: str, q_emb: Optional[List[float]],
                           exclude_ids: Set[int]):
        """One structure-similar / surface-different memory, or None."""
        if not self._hcfg.get("analogy_slot", True) or not _HRR_OK:
            return None
        min_chars = int(self._hcfg.get("analogy_min_query_chars", 40) or 0)
        if len(rolling_query or "") < min_chars:
            return None
        got = self._hrr_matrix()
        if not got or got[0] is None:
            return None
        ids_arr, mat, contents = got
        try:
            dim = mat.shape[1]
            # Fact vectors are role-bound: encode_fact bundles
            # bind(encode_text(content), role_content) with entity bindings.
            # bind is elementwise phase addition, so probing with the SAME
            # role binding preserves content similarity exactly:
            # sim(bind(q, r), bind(c, r)) == sim(q, c). Probe the content
            # channel with distinctive words only (stopwords add shared noise).
            probe_text = _content_words(rolling_query) or rolling_query
            q = _hrrmod.bind(
                _hrrmod.encode_text(probe_text, dim),
                _hrrmod.encode_atom("__hrr_role_content__", dim))
            sims = (_np.mean(_np.cos(mat - q), axis=1) + 1.0) / 2.0
        except Exception as e:
            logger.debug("hybrid: analogy HRR compute failed (%s)", e)
            return None
        hrr_min = float(self._hcfg.get("analogy_hrr_min", 0.50))
        z_min = float(self._hcfg.get("analogy_z_min", 2.0))
        surface_max = float(self._hcfg.get("analogy_surface_max", 0.30))
        mu = float(_np.mean(sims))
        sd = float(_np.std(sims))
        # Relative gate: the candidate must stand out from the whole store's
        # similarity distribution, not just clear an absolute bar.
        gate = max(hrr_min, mu + z_min * sd) if sd > 1e-9 else hrr_min
        skip = set(exclude_ids) | self._recent_analogy_ids()
        order = _np.argsort(-sims)
        q_toks = _tokens(rolling_query)
        best = None
        checked = 0
        for i in order:
            if sims[i] < gate or checked >= 8:
                break
            fid = int(ids_arr[i])
            if fid in skip:
                continue
            checked += 1
            surf = self._surface_similarity(fid, contents.get(fid, ""), q_emb, q_toks)
            if surf > surface_max:
                continue
            score = float(sims[i]) - surf
            if best is None or score > best[0]:
                best = (score, fid, float(sims[i]))
        if best is None:
            return None
        _, fid, hrr_sim = best
        content = (contents.get(fid, "") or "").strip()
        if len(content) > 220:
            content = content[:217] + "..."
        return {"fact_id": fid, "vid": f"holo_fact_{fid}", "score": round(hrr_sim, 3),
                "block": "analogy",
                "line": f"- [~{hrr_sim:.2f}] {content}"}

    # -- situations ------------------------------------------------------------

    def _situation_matches(self, q_emb: Optional[List[float]],
                           rolling_query: str) -> List[Dict[str, Any]]:
        if q_emb is None or self._memcon is None or _ms is None or not self._graph:
            return []
        sim_min = float(self._hcfg.get("situation_sim_min", 0.42))
        try:
            raw = _ms.query_vectors(self._memcon, embedding=q_emb, k=24)
        except Exception:
            return []
        out = []
        for item in raw:
            if (item.get("type") or "") != "situation":
                continue
            if item["similarity"] < sim_min:
                continue
            name = (item.get("text") or "").split(":", 1)[0].strip()
            node = name if name in self._graph else self._resolve_node(name)
            playbook = ""
            if node is not None:
                playbook = str(self._graph.nodes[node].get("playbook", "") or "")
            line = f"- {name}: {playbook}" if playbook else f"- {item.get('text','')}"
            out.append({"fact_id": None, "vid": item.get("vid", ""),
                        "score": item["similarity"], "block": "situation",
                        "line": line[:300]})
            if len(out) >= 2:
                break
        return out

    # -- prompt + prefetch ---------------------------------------------------

    def system_prompt_block(self) -> str:
        base = super().system_prompt_block() if _HOLO else ""
        extra = (
            "# Semantic + Graph Memory\n"
            "Vector recall and a knowledge graph back the fact store. Relevant memories are "
            "auto-surfaced each turn (email/lead exhaust excluded; reachable via tools). "
            "Tools: `recall` (fuzzy semantic lookup), `graph_query` (an entity's direct "
            "relationships), `graph_path` (how two entities connect), `graph_connections` "
            "(how several relate / shared traits), `graph_reason` (multi-hop reasoning over "
            "a subgraph), `chunk_expand` (unpack a summarized chunk memory), "
            "`situation_store` (name a recurring situation + playbook), `analogize` "
            "(find structurally similar entities/situations).\n"
            "An auto-injected '## Analogy candidate' line is a speculative reminding, "
            "structurally similar to the current situation but surface-different. Treat it "
            "as a hint, not a fact about the current topic. When an injected memory proves "
            "genuinely useful or wrong, rate it with fact_feedback (trains recall)."
        )
        return (base + "\n\n" + extra).strip() if base else extra

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        entries: List[Dict[str, Any]] = []
        fts_norm: Set[str] = set()
        self._push_rolling(query)
        rolling_query = "\n".join(self._rolling) if self._rolling else (query or "")
        ops = self._op_categories()

        # 1) holographic FTS recall (fast, keyword/entity), operational-filtered,
        #    with fact ids retained for counting.
        fts_lines: List[str] = []
        retriever = getattr(self, "_retriever", None) if _HOLO else None
        if retriever is not None and query:
            try:
                results = retriever.search(query, min_trust=getattr(self, "_min_trust", 0.3),
                                           limit=12) or []
            except Exception as e:
                logger.debug("hybrid: FTS search failed (%s)", e)
                results = []
            for r in results:
                if (r.get("category") or "") in ops:
                    continue
                content = r.get("content", "")
                trust = r.get("trust_score", r.get("trust", 0)) or 0
                line = f"- [{trust:.1f}] {content}"
                fts_lines.append(line)
                fts_norm.add(_norm(line))
                fid = r.get("fact_id")
                entries.append({"fact_id": fid, "vid": f"holo_fact_{fid}" if fid else None,
                                "score": r.get("score"), "block": "fts", "line": line,
                                "category": r.get("category")})
                if len(fts_lines) >= 5:
                    break
        elif _HOLO:
            # Degraded mode: base internals moved, fall back to the bundled
            # text-only prefetch (no counting, no filtering).
            try:
                fts = super().prefetch(query, session_id=session_id)
            except Exception:
                fts = ""
            if fts:
                fts_lines = [l for l in fts.splitlines() if not l.startswith("#")]
                for line in fts_lines:
                    fts_norm.add(_norm(line))

        fts_block = ("## Holographic Memory\n" + "\n".join(fts_lines)) if fts_lines else ""

        # Shared query embedding (vector block, situations, analogy surface check).
        q_emb = None
        if self._vector_ok and _ms is not None and rolling_query:
            try:
                q_emb = _ms.embed([rolling_query])[0]
            except Exception as e:
                logger.debug("hybrid: query embed failed (%s)", e)

        # 2) semantic vector recall (deduped against FTS so facts don't repeat).
        # Situations are excluded here, they surface via the dedicated block
        # below, which carries the playbook instead of the raw description.
        vlines, seen = [], set()
        vec_entries: List[Dict[str, Any]] = []
        for v in self._vector_search(rolling_query, k=6,
                                     exclude_types=self._op_vec_types() | {"situation"},
                                     embedding=q_emb):
            if v["similarity"] < PREFETCH_SIM_THRESHOLD:
                continue
            n = _norm(v["text"])
            if n in seen or any(n in f or f in n for f in fts_norm):
                continue
            seen.add(n)
            line = f"- [{v['similarity']:.2f}] {v['text']}"
            vlines.append(line)
            fid = self._fact_id_from_vid(v.get("vid", ""))
            e = {"fact_id": fid, "vid": v.get("vid"), "score": v["similarity"],
                 "block": "vector", "line": line}
            entries.append(e)
            vec_entries.append(e)
            if len(vlines) >= 5:
                break
        vec_block = ("## Semantic Recall\n" + "\n".join(vlines)) if vlines else ""

        # Chunk awareness: when a chunk and one of its members both surfaced,
        # keep the chunk, drop the member line (unpack on demand via chunk_expand).
        try:
            surfaced = {e["fact_id"] for e in entries if e.get("fact_id")}
            if surfaced and self._memcon is not None:
                qmarks = ",".join("?" * len(surfaced))
                rows = self._memcon.execute(
                    f"SELECT chunk_fact_id, member_fact_id FROM chunk_members "
                    f"WHERE chunk_fact_id IN ({qmarks}) AND member_fact_id IN ({qmarks})",
                    list(surfaced) + list(surfaced)).fetchall()
                drop = {int(r[1]) for r in rows}
                if drop:
                    had_fts_entries = any(e["block"] == "fts" for e in entries)
                    entries = [e for e in entries if e.get("fact_id") not in drop]
                    # Rebuild both blocks from the surviving entries (skip the
                    # FTS rebuild in degraded mode, where FTS lines are opaque).
                    if had_fts_entries:
                        fts_keep = [e["line"] for e in entries if e["block"] == "fts"]
                        fts_block = ("## Holographic Memory\n" + "\n".join(fts_keep)) if fts_keep else ""
                    vec_keep = [e["line"] for e in entries if e["block"] == "vector"]
                    vec_block = ("## Semantic Recall\n" + "\n".join(vec_keep)) if vec_keep else ""
        except Exception as e:
            logger.debug("hybrid: chunk suppression failed (%s)", e)

        # 3) graph neighborhood for entities named in the query (alias-aware)
        graph_block = self._graph_context(query or rolling_query)

        # 4) analogy slot, structure-similar, surface-different (max 1 line)
        analogy_block = ""
        try:
            cand = self._analogy_candidate(
                rolling_query, q_emb,
                exclude_ids={e["fact_id"] for e in entries if e.get("fact_id")})
        except Exception as e:
            logger.debug("hybrid: analogy slot failed (%s)", e)
            cand = None
        if cand:
            analogy_block = "## Analogy candidate\n" + cand["line"]
            entries.append(cand)

        # 5) situation matches (max 2 lines)
        situation_block = ""
        try:
            sits = self._situation_matches(q_emb, rolling_query)
        except Exception as e:
            logger.debug("hybrid: situation match failed (%s)", e)
            sits = []
        if sits:
            situation_block = "## Situation match\n" + "\n".join(s["line"] for s in sits)
            entries.extend(sits)

        # Assemble with a hard char cap; count/log only what actually shipped.
        cap = int(self._hcfg.get("prefetch_char_cap", 2000) or 0)
        text, included = _cap_blocks(
            [fts_block, vec_block, graph_block, analogy_block, situation_block], cap)
        shipped = [e for e in entries if e.get("line") in included]
        self._bump_retrieval([e["fact_id"] for e in shipped if e.get("fact_id")])
        self._log_recall(query or rolling_query, shipped)
        return text

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        base = (super().get_tool_schemas() if _HOLO else []) or []
        schemas = list(base) + [RECALL_SCHEMA, GRAPH_QUERY_SCHEMA, GRAPH_PATH_SCHEMA,
                                GRAPH_CONNECTIONS_SCHEMA, GRAPH_REASON_SCHEMA,
                                CHUNK_EXPAND_SCHEMA, SITUATION_STORE_SCHEMA,
                                DECISION_LOG_SCHEMA, ONTOLOGY_REVIEW_SCHEMA]
        if self._hcfg.get("analogize_enabled", True):
            schemas.append(ANALOGIZE_SCHEMA)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "recall":
            return self._handle_recall(args)
        if tool_name == "graph_query":
            return self._handle_graph_query(args)
        if tool_name == "graph_path":
            return self._handle_graph_path(args)
        if tool_name == "graph_connections":
            return self._handle_graph_connections(args)
        if tool_name == "graph_reason":
            return self._handle_graph_reason(args)
        if tool_name == "chunk_expand":
            return self._handle_chunk_expand(args)
        if tool_name == "situation_store":
            return self._handle_situation_store(args)
        if tool_name == "decision_log":
            return self._handle_decision_log(args)
        if tool_name == "ontology_review":
            return self._handle_ontology_review(args)
        if tool_name == "analogize":
            return self._handle_analogize(args)
        if _HOLO:
            result = super().handle_tool_call(tool_name, args, **kwargs)
            if tool_name == "fact_store" and args.get("action") in (
                    "search", "probe", "related", "reason"):
                self._count_tool_results(result, block="fact_store",
                                         query=args.get("query") or args.get("entity")
                                         or ",".join(args.get("entities") or []))
            return result
        return tool_error(f"Unknown tool: {tool_name}")

    def _count_tool_results(self, result_json: str, *, block: str, query: str = "") -> None:
        try:
            data = json.loads(result_json)
            results = data.get("results") or []
            entries = []
            for r in results:
                fid = r.get("fact_id") if isinstance(r, dict) else None
                if fid is None:
                    continue
                entries.append({"fact_id": int(fid), "vid": f"holo_fact_{fid}",
                                "score": r.get("score"), "block": block})
            if entries:
                self._bump_retrieval([e["fact_id"] for e in entries])
                self._log_recall(query or "", entries)
        except Exception as e:
            logger.debug("hybrid: tool result counting failed (%s)", e)

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("recall requires 'query'")
        if not self._vector_ok:
            return tool_error("vector recall unavailable; use fact_store instead")
        k = max(1, min(int(args.get("k", 5) or 5), 15))
        include_op = args.get("include_operational", True)
        exclude = None if include_op else self._op_vec_types()
        results = [r for r in self._vector_search(query, k=k, exclude_types=exclude)
                   if r["similarity"] >= RECALL_SIM_FLOOR]
        entries = []
        for r in results:
            fid = self._fact_id_from_vid(r.get("vid", ""))
            entries.append({"fact_id": fid, "vid": r.get("vid"),
                            "score": r["similarity"], "block": "recall_tool"})
        self._bump_retrieval([e["fact_id"] for e in entries if e.get("fact_id")])
        self._log_recall(query, entries)
        slim = [{"text": r["text"], "source": r["source"], "similarity": r["similarity"]}
                for r in results]
        return json.dumps({"results": slim, "count": len(slim)})

    def _handle_graph_query(self, args: Dict[str, Any]) -> str:
        entity = args.get("entity", "")
        if not entity:
            return tool_error("graph_query requires 'entity'")
        if not self._graph:
            return json.dumps({"error": "knowledge graph unavailable", "connections": []})
        depth = max(1, min(int(args.get("depth", 1) or 1), 2))
        node = self._resolve_node(entity)
        if node is None:
            cands = [str(n) for n in self._graph.nodes() if entity.lower() in str(n).lower()][:5]
            return json.dumps({"error": f"'{entity}' not found", "did_you_mean": cands, "connections": []})
        conns = self._neighbors(node, depth=depth)
        return json.dumps({
            "entity": node,
            "node": {k: v for k, v in dict(self._graph.nodes[node]).items() if isinstance(v, (str, int, float, bool))},
            "connections": conns,
            "count": len(conns),
        })

    def _candidates(self, name: str, limit: int = 5) -> list:
        if not self._graph or not name:
            return []
        nl = name.lower()
        return [str(n) for n in self._graph.nodes() if nl in str(n).lower()][:limit]

    def _handle_graph_path(self, args: Dict[str, Any]) -> str:
        a, b = args.get("source", ""), args.get("target", "")
        if not a or not b:
            return tool_error("graph_path requires 'source' and 'target'")
        if not self._graph:
            return json.dumps({"error": "knowledge graph unavailable"})
        import networkx as nx
        na, nb = self._resolve_node(a), self._resolve_node(b)
        if na is None:
            return json.dumps({"error": f"'{a}' not found", "did_you_mean": self._candidates(a)})
        if nb is None:
            return json.dumps({"error": f"'{b}' not found", "did_you_mean": self._candidates(b)})
        U = self._graph.to_undirected(as_view=True)
        try:
            path = nx.shortest_path(U, na, nb)
        except nx.NetworkXNoPath:
            return json.dumps({"connected": False, "source": na, "target": nb})
        except Exception as e:
            return json.dumps({"error": str(e)})
        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            rel = (self._graph.edges[u, v].get("relation") if self._graph.has_edge(u, v)
                   else self._graph.edges[v, u].get("relation", "RELATED_TO"))
            steps.append({"from": u, "relation": rel, "to": v})
        max_hops = int(args.get("max_hops", 6) or 6)
        return json.dumps({"connected": True, "hops": len(path) - 1,
                           "path": steps, "over_max_hops": (len(path) - 1) > max_hops})

    def _handle_graph_connections(self, args: Dict[str, Any]) -> str:
        ents = args.get("entities") or []
        if not isinstance(ents, list) or len(ents) < 2:
            return tool_error("graph_connections requires 'entities' (a list of 2+)")
        if not self._graph:
            return json.dumps({"error": "knowledge graph unavailable"})
        import networkx as nx
        U = self._graph.to_undirected(as_view=True)
        resolved, missing = [], []
        for e in ents:
            n = self._resolve_node(e)
            resolved.append(n) if n else missing.append(e)
        out = {"resolved": resolved, "missing": missing, "pairwise_paths": [], "shared_neighbors": []}
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                try:
                    p = nx.shortest_path(U, resolved[i], resolved[j])
                    out["pairwise_paths"].append(
                        {"from": resolved[i], "to": resolved[j], "hops": len(p) - 1, "via": p[1:-1]})
                except Exception:
                    pass
        if len(resolved) >= 2:
            sets = [set(U.neighbors(n)) for n in resolved]
            shared = set.intersection(*sets) if sets else set()
            out["shared_neighbors"] = sorted(str(x) for x in shared)[:15]
        return json.dumps(out)

    def _handle_graph_reason(self, args: Dict[str, Any]) -> str:
        q = args.get("question", "")
        if not q:
            return tool_error("graph_reason requires 'question'")
        if not self._graph:
            return json.dumps({"error": "knowledge graph unavailable"})
        seeds = [self._resolve_node(e) for e in (args.get("entities") or [])]
        seeds = [s for s in seeds if s]
        if not seeds:
            ql = q.lower()
            seeds = [n for n in self._graph.nodes() if len(str(n)) >= 4 and str(n).lower() in ql][:4]
        if not seeds:
            return json.dumps({"error": "no entities matched; pass 'entities' explicitly"})
        keep = set(seeds)
        for s in seeds:
            nbrs = set(self._graph.successors(s)) | set(self._graph.predecessors(s))
            keep |= nbrs
            for nb in list(nbrs)[:20]:
                keep |= set(self._graph.successors(nb)) | set(self._graph.predecessors(nb))
        keep = set(list(keep)[:140])
        lines = [f"{u} {d.get('relation', 'RELATED_TO')} {v}"
                 for u, v, d in self._graph.edges(data=True) if u in keep and v in keep]
        if not lines:
            return json.dumps({"error": "empty subgraph around those entities"})
        answer = self._llm_reason(q, "\n".join(lines[:220]))
        return json.dumps({"answer": answer, "seeds": seeds, "subgraph_edges": len(lines)})

    def _handle_chunk_expand(self, args: Dict[str, Any]) -> str:
        try:
            chunk_id = int(args["chunk_fact_id"])
        except Exception:
            return tool_error("chunk_expand requires integer 'chunk_fact_id'")
        store = getattr(self, "_store", None)
        conn = getattr(store, "_conn", None)
        if conn is None or self._memcon is None:
            return tool_error("chunk store unavailable")
        try:
            member_ids = [int(r[0]) for r in self._memcon.execute(
                "SELECT member_fact_id FROM chunk_members WHERE chunk_fact_id=?",
                (chunk_id,)).fetchall()]
            if not member_ids:
                return json.dumps({"chunk_fact_id": chunk_id, "members": [],
                                   "note": "no members recorded for this chunk"})
            qmarks = ",".join("?" * len(member_ids))
            lock = getattr(store, "_lock", None)
            def _q():
                return conn.execute(
                    f"SELECT fact_id, content, category, trust_score FROM facts "
                    f"WHERE fact_id IN ({qmarks})", member_ids).fetchall()
            rows = None
            if lock is not None:
                with lock:
                    rows = _q()
            else:
                rows = _q()
            members = [{"fact_id": r[0], "content": r[1], "category": r[2],
                        "trust": r[3]} for r in rows]
            self._bump_retrieval(member_ids)
            self._log_recall(f"chunk_expand:{chunk_id}",
                             [{"fact_id": m["fact_id"], "vid": f"holo_fact_{m['fact_id']}",
                               "score": None, "block": "fact_store"} for m in members])
            return json.dumps({"chunk_fact_id": chunk_id, "members": members,
                               "count": len(members)})
        except Exception as e:
            return tool_error(str(e))

    def _handle_decision_log(self, args: Dict[str, Any]) -> str:
        action = args.get("action", "")
        if self._memcon is None:
            return tool_error("decision log unavailable")
        sid = getattr(self, "_session_id", "") or ""
        if action == "record":
            proposal = (args.get("proposal") or "").strip()
            if not proposal:
                return tool_error("decision_log record requires 'proposal'")
            kind = args.get("kind") or "recommendation"
            opts = args.get("options_shown")
            opts_json = json.dumps([str(o)[:200] for o in opts][:10],
                                   ensure_ascii=False) if isinstance(opts, list) else "[]"
            # Auto-attach what memory was surfaced this turn: these are the
            # facts the recommendation was (potentially) based on.
            refs = []
            try:
                refs = [{"fact_id": r[0], "vid": r[1]} for r in self._memcon.execute(
                    "SELECT fact_id, vid FROM recall_log "
                    "WHERE session_id=? AND turn_number=? LIMIT 20",
                    (sid, self._turn_number))]
            except Exception:
                pass
            try:
                cur = self._memcon.execute(
                    "INSERT INTO decision_log(session_id, turn_number, kind, proposal, "
                    "options_shown, source_refs) VALUES (?,?,?,?,?,?)",
                    (sid, self._turn_number, kind, proposal[:500], opts_json,
                     json.dumps(refs, ensure_ascii=False)))
                self._memcon.commit()
                return json.dumps({"status": "recorded", "decision_id": cur.lastrowid,
                                   "linked_recalls": len(refs)})
            except Exception as e:
                return tool_error(f"decision record failed: {e}")
        if action == "resolve":
            outcome = args.get("outcome") or ""
            if outcome not in ("accepted", "rejected", "modified"):
                return tool_error("decision_log resolve requires outcome in "
                                  "accepted|rejected|modified")
            did = args.get("decision_id")
            try:
                if did is None:
                    row = self._memcon.execute(
                        "SELECT id FROM decision_log WHERE outcome='pending' "
                        "AND session_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
                    if row is None:
                        # Cross-session fallback: user may reply in a later session.
                        row = self._memcon.execute(
                            "SELECT id FROM decision_log WHERE outcome='pending' "
                            "ORDER BY id DESC LIMIT 1").fetchone()
                    if row is None:
                        return tool_error("no pending decision to resolve")
                    did = row[0]
                n = self._memcon.execute(
                    "UPDATE decision_log SET outcome=?, chosen=?, reason=? "
                    "WHERE id=? AND outcome='pending'",
                    (outcome, (args.get("chosen") or "")[:500],
                     (args.get("reason") or "")[:500], int(did))).rowcount
                self._memcon.commit()
                if not n:
                    return tool_error(f"decision {did} not found or already resolved")
                return json.dumps({"status": "resolved", "decision_id": int(did),
                                   "outcome": outcome})
            except Exception as e:
                return tool_error(f"decision resolve failed: {e}")
        if action == "list":
            try:
                rows = self._memcon.execute(
                    "SELECT id, ts, kind, proposal, outcome, chosen FROM decision_log "
                    "ORDER BY id DESC LIMIT 15").fetchall()
                return json.dumps({"decisions": [
                    {"id": r[0], "ts": r[1], "kind": r[2], "proposal": r[3],
                     "outcome": r[4], "chosen": r[5] or ""} for r in rows]})
            except Exception as e:
                return tool_error(str(e))
        return tool_error("decision_log requires action record|resolve|list")

    def _handle_ontology_review(self, args: Dict[str, Any]) -> str:
        action = args.get("action", "")
        if self._memcon is None:
            return tool_error("ontology review unavailable")
        try:
            import ontology_review as orv
        except Exception as e:
            return tool_error(f"ontology_review module unavailable: {e}")
        con = self._memcon
        if action == "list":
            return json.dumps({
                "aliases": [{"a": a, "b": b} for a, b, _ in orv.pending_aliases(con)],
                "categories": [{"label": k, "members": v}
                               for k, v in orv.pending_categories(con).items()],
            })
        if action in ("approve", "reject"):
            kind = args.get("kind") or ""
            try:
                if kind == "alias":
                    a, b = (args.get("a") or "").strip(), (args.get("b") or "").strip()
                    if not a or not b:
                        return tool_error("alias review requires 'a' (merge away) and 'b' (keep)")
                    result = (orv.approve_alias(con, a, b) if action == "approve"
                              else orv.reject_alias(con, a, b))
                elif kind == "category":
                    label = (args.get("label") or "").strip()
                    if not label:
                        return tool_error("category review requires 'label'")
                    result = (orv.approve_category(con, label) if action == "approve"
                              else orv.reject_category(con, label))
                else:
                    return tool_error("ontology_review requires kind alias|category")
            except Exception as e:
                return tool_error(f"ontology review failed: {e}")
            if action == "approve":
                self._load_graph()  # refresh the cached graph after mutations
            return json.dumps(result)
        return tool_error("ontology_review requires action list|approve|reject")

    def _handle_situation_store(self, args: Dict[str, Any]) -> str:
        action = args.get("action", "")
        if self._memcon is None or _ms is None:
            return tool_error("situation store unavailable")
        if action == "add":
            name = (args.get("name") or "").strip()
            desc = (args.get("description") or "").strip()
            playbook = (args.get("playbook") or "").strip()
            if not name or not desc or not playbook:
                return tool_error("situation add requires 'name', 'description', 'playbook'")
            try:
                _ms.add_node(self._memcon, name, "situation",
                             {"description": desc, "playbook": playbook})
                vid = f"situation_{_hash(name.lower())}"
                _ms.upsert_vector(self._memcon, vid, f"{name}: {desc}",
                                  "situation_store", "situation",
                                  embedding=_ms.embed([desc])[0])
                self._memcon.commit()
                self._load_graph()
                return json.dumps({"status": "added", "name": name, "vid": vid})
            except Exception as e:
                return tool_error(f"situation add failed: {e}")
        if action == "list":
            try:
                rows = self._memcon.execute(
                    "SELECT name, attrs FROM graph_nodes WHERE type='situation'").fetchall()
                sits = []
                for name, attrs in rows:
                    a = {}
                    try:
                        a = json.loads(attrs) if attrs else {}
                    except Exception:
                        pass
                    sits.append({"name": name, "description": a.get("description", ""),
                                 "playbook": a.get("playbook", "")})
                return json.dumps({"situations": sits, "count": len(sits)})
            except Exception as e:
                return tool_error(str(e))
        if action == "remove":
            name = (args.get("name") or "").strip()
            if not name:
                return tool_error("situation remove requires 'name'")
            try:
                self._memcon.execute(
                    "DELETE FROM graph_nodes WHERE name=? AND type='situation'", (name,))
                _ms.delete_vector(self._memcon, f"situation_{_hash(name.lower())}")
                self._memcon.commit()
                self._load_graph()
                return json.dumps({"status": "removed", "name": name})
            except Exception as e:
                return tool_error(str(e))
        return tool_error(f"Unknown situation_store action: {action}")

    # -- analogize -------------------------------------------------------------

    def _fingerprint(self, node) -> collections.Counter:
        fp = collections.Counter()
        try:
            for _, nb, d in self._graph.out_edges(node, data=True):
                fp[("out", d.get("relation", "RELATED_TO"),
                    self._graph.nodes[nb].get("node_type", "concept"))] += 1
            for nb, _, d in self._graph.in_edges(node, data=True):
                fp[("in", d.get("relation", "RELATED_TO"),
                    self._graph.nodes[nb].get("node_type", "concept"))] += 1
        except Exception:
            pass
        return fp

    @staticmethod
    def _counter_cosine(a: collections.Counter, b: collections.Counter) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _text_seed_entities(self, text: str, limit: int = 3) -> List[str]:
        """Seed entities for free-text analogize: top gist/fact vector hits →
        their facts' linked entities that exist as graph nodes."""
        if self._memcon is None or _ms is None:
            return []
        store = getattr(self, "_store", None)
        conn = getattr(store, "_conn", None)
        seeds: List[str] = []
        try:
            raw = _ms.query_vectors(self._memcon, query_text=text, k=12)
            fids = [self._fact_id_from_vid(r.get("vid", "")) for r in raw]
            fids = [f for f in fids if f is not None][:6]
            if conn is not None and fids:
                qmarks = ",".join("?" * len(fids))
                rows = conn.execute(
                    f"SELECT DISTINCT e.name FROM fact_entities fe "
                    f"JOIN entities e ON e.entity_id = fe.entity_id "
                    f"WHERE fe.fact_id IN ({qmarks})", fids).fetchall()
                for (name,) in rows:
                    node = self._resolve_node(name)
                    if node is not None and node not in seeds:
                        seeds.append(node)
                    if len(seeds) >= limit:
                        break
        except Exception as e:
            logger.debug("hybrid: analogize text seeding failed (%s)", e)
        return seeds

    def _handle_analogize(self, args: Dict[str, Any]) -> str:
        if not self._hcfg.get("analogize_enabled", True):
            return tool_error("analogize disabled by config")
        if not self._graph:
            return json.dumps({"error": "knowledge graph unavailable"})
        entity = (args.get("entity") or "").strip()
        text = (args.get("text") or "").strip()
        k = max(1, min(int(args.get("k", 5) or 5), 10))
        sources: List[str] = []
        if entity:
            node = self._resolve_node(entity)
            if node is None:
                return json.dumps({"error": f"'{entity}' not found",
                                   "did_you_mean": self._candidates(entity)})
            sources = [node]
        elif text:
            sources = self._text_seed_entities(text)
            if not sources:
                return json.dumps({"error": "no graph entities matched that description; "
                                            "try analogize with an explicit 'entity'"})
        else:
            return tool_error("analogize requires 'entity' or 'text'")

        mappings = []
        for src in sources:
            src_fp = self._fingerprint(src)
            if not src_fp:
                continue
            neighbors = set(self._graph.successors(src)) | set(self._graph.predecessors(src))
            src_norm = _er_normalize(str(src))
            for cand in self._graph.nodes():
                if cand == src or cand in neighbors:
                    continue
                cand_norm = _er_normalize(str(cand))
                if not cand_norm or not src_norm:
                    continue
                # Surface-different only: skip near-identical names.
                if (cand_norm in src_norm or src_norm in cand_norm or
                        difflib.SequenceMatcher(None, src_norm, cand_norm).ratio() >= 0.7):
                    continue
                cand_fp = self._fingerprint(cand)
                if not cand_fp:
                    continue
                sim = self._counter_cosine(src_fp, cand_fp)
                if sim <= 0.0:
                    continue
                shared = sorted({rel for (_, rel, _) in (set(src_fp) & set(cand_fp))})
                mappings.append({"source": str(src), "target": str(cand),
                                 "similarity": round(sim, 3),
                                 "shared_relations": shared[:8]})
        mappings.sort(key=lambda m: -m["similarity"])
        mappings = mappings[:k]
        out: Dict[str, Any] = {"mappings": mappings, "count": len(mappings)}
        if args.get("explain") and mappings:
            best = mappings[0]
            src_edges = [f"{c['from']} {c['relation']} {c['to']}"
                         for c in self._neighbors(best["source"], depth=1)[:12]]
            tgt_edges = [f"{c['from']} {c['relation']} {c['to']}"
                         for c in self._neighbors(best["target"], depth=1)[:12]]
            prompt = (f"SOURCE ({best['source']}):\n" + "\n".join(src_edges) +
                      f"\n\nTARGET ({best['target']}):\n" + "\n".join(tgt_edges))
            out["explanation"] = self._llm_reason(
                f"Explain the structural analogy between {best['source']} and "
                f"{best['target']}: what plays what role, and what does the mapping "
                f"suggest or predict?", prompt)
        return json.dumps(out)

    def _llm_reason(self, question: str, subgraph: str) -> str:
        key = _read_env_key("OPENROUTER_API_KEY")
        if not key:
            return "(graph_reason needs OPENROUTER_API_KEY)"
        import urllib.request
        system = (
            "You are a reasoning engine over a knowledge graph. You are given relationship "
            "triples ('subject RELATION object') and a question. Reason over them, including "
            "multi-hop chains, and answer concisely and concretely, citing the relationship "
            "path(s) you used. If the graph doesn't support an answer, say so plainly."
        )
        payload = json.dumps({
            "model": REASON_MODEL, "temperature": 0.2, "max_tokens": 700,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"GRAPH:\n{subgraph}\n\nQUESTION: {question}"},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(reasoning failed: {e})"

    # -- instant sync --------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        # keep holographic's mirror-to-facts behavior
        if _HOLO:
            try:
                super().on_memory_write(action, target, content)
            except Exception as e:
                logger.debug("hybrid: holographic mirror failed (%s)", e)
        # instant vector upsert so semantic recall reflects edits immediately
        if not self._vector_ok or not content or self._memcon is None or _ms is None:
            return
        try:
            text = content.strip()
            is_user = target == "user"
            vid = f"{'user_' if is_user else 'memory_'}{_hash(text)}"
            src = "USER.md" if is_user else "MEMORY.md"
            vtype = "user_profile" if is_user else "agent_note"
            if action in ("add", "replace"):
                _ms.upsert_vector(self._memcon, vid, text, src, vtype)
                self._memcon.commit()
            elif action == "remove":
                _ms.delete_vector(self._memcon, vid)
                self._memcon.commit()
        except Exception as e:
            logger.debug("hybrid: vector sync on memory_write failed (%s)", e)

    def shutdown(self) -> None:
        try:
            if self._memcon is not None:
                self._memcon.close()
        except Exception:
            pass
        self._memcon = None
        self._hrr_cache = None
        if _HOLO:
            try:
                super().shutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the hybrid memory provider with the plugin system."""
    ctx.register_memory_provider(HybridMemoryProvider())
