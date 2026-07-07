# Architecture

## The three layers, one database

`memory_store.db` (SQLite) holds three logically separate layers that share
one file so there's never a sync problem between them:

1. **Facts**: `facts`, `facts_fts` (FTS5), `entities`, `fact_entities`. Owned
   by the upstream holographic provider (`plugins/memory/holographic` in
   Hermes Agent). Each fact has a `trust_score`, `retrieval_count`,
   `helpful_count`, and an HRR phase-vector encoding (`hrr_vector`) used for
   the analogy slot. This layer is never touched directly by the code in this
   repo, only read from and (for trust/retrieval bookkeeping) updated via
   its own connection.
2. **Vectors**: `vec_items` + `vec_index` (a `sqlite-vec` virtual table,
   cosine distance, 384-dim). Owned by `memstore.py`. Every fact, every
   `MEMORY.md`/`USER.md` section, every gist, every situation description
   gets a vector row keyed by a stable `vid` (`holo_fact_<id>`, `gist_<id>`,
   `memory_<hash>`, `user_<hash>`, `situation_<hash>`, ...).
3. **Graph**: `graph_nodes` + `edges` (plain tables, also owned by
   `memstore.py`). Loaded into a `networkx.DiGraph` on demand for path/reason
   queries; never kept resident. Edges carry a `src_tag` (`ingest`,
   `calendar`, `chunk`, `proposed`, `alias-candidate`, ...) so any batch of
   writes can be selectively deleted and redone (`remove_edges_by_tag`).

Nothing here does a periodic full rebuild. Every writer is incremental and
transactional against the same connection pattern (`memstore.connect()` →
`init_schema()` → write → `commit()`), which is what makes the whole system
cheap enough to run entirely on a 2 GB box.

## Data flow

### 1. Ingest: documents become facts + graph

`memory_ingest.py` is the single entry point for turning unstructured text
(an email, a call transcript, a note) into memory. It sends the document to
an LLM (`gemini-2.5-flash-lite` by default, cheap and fast; this runs on
every spooled document) with a system prompt asking for JSONL output: facts,
nodes, and edges, each on its own line. Notably it also asks for **dated
event nodes** (`{"node": {"name": "Trip to Prague (2026-08-13)", "type":
"event", "attrs": {"start": ..., "location": ...}}}`). Occurrences get
first-class graph citizenship instead of being buried in prose, with the
start date embedded in the node name so recurring/similar events stay
distinct.

The node-type enum the prompt is allowed to emit is a fixed whitelist
(`person`, `company`, `project`, `idea`, `location`, `event`, `job`,
`document`); anything else is dropped rather than trusted, so the graph's type
space stays closed the same way its relation space does.

Raw document text is *never* stored as a fact, only the distilled output.
An ingest ledger (`~/.hermes/ingest/ledger.json`) makes re-runs idempotent.
Every fact and every edge also carries a `source_ref` stamped at write time
(`<source>:<docid>` for ingest, `gcal:<eventId>` for calendar), so a surfaced
memory can be traced back to the email, call, or event it came from; see
[Provenance](#11-provenance) below.

Every node/edge from extraction passes through `entity_resolve.py` before
being written (see below), and lands directly in `memstore`; there's no
intermediate overlay file to periodically merge.

### 2. Entity resolution: one canonical node per real-world entity

`entity_resolve.py` maintains a persistent index
(`knowledge_graph/entity_index.json`) mapping merge keys to canonical display
names. Resolution order, strongest signal first:

1. Exact merge-key hit: `domain:<d>` → `email:<e>` → `phone:<last10>` → `norm:<normalized-name>`
2. Fuzzy fallback on normalized name (`difflib.SequenceMatcher`, same-type
   candidates only, first-character bucketed for cost, and, critically, a
   **distinct-domain guard**: two organizations are never fuzzy-merged if
   they carry different domains, even if the names look similar). Above
   `FUZZY_THRESHOLD` (0.92) it merges silently; between 0.85 and 0.92 it's
   recorded as a `pending_alias` and written as a reviewable
   `POSSIBLE_ALIAS` edge instead of a silent merge.
3. Otherwise: register a new canonical entity.

A **closed** controlled vocabulary (`canon_relation`) collapses ad-hoc LLM
relation labels (`works_at`, `employed_by`, `staff_of`, ...) onto one
canonical predicate (`WORKS_FOR`) so multi-hop graph queries, relational
fingerprints, and analogy search all stay consistent. Resolution runs in a
fixed order: exact map, then an inverse/passive map (`owned_by` -> `OWNS` with
the edge direction flipped, signalled by a `flipped=True` return the caller
acts on), then an ordered set of stem rules, and finally a `RELATED_TO`
default. Crucially there is **no passthrough**: an unrecognized label lands on
`RELATED_TO`, it is never minted as a new relation type. (An earlier version
did pass novel relations through cleaned; in practice that grew the graph to
hundreds of distinct predicates from a few thousand edges, which broke
fingerprint comparison, so the vocabulary was closed to a fixed set of ~38
canonical predicates.) The raw extractor string is preserved in
`edges.rel_orig` whenever `canon_relation` changed it, so vocabulary drift is
observable (`SELECT rel_orig, COUNT(*) FROM edges WHERE rel_orig != ''`).
`normalize_relation` remains as a thin wrapper returning just the canonical
string.

### 3. Prefetch: what gets injected every turn

`HybridMemoryProvider.prefetch()` runs on every user turn and assembles up to
five blocks, each independently sourced, then caps the total to
`prefetch_char_cap` characters (default 2000) by walking blocks in priority
order and keeping whole lines:

1. **Holographic FTS**: fast keyword/entity search over facts (from the
   upstream provider), operational categories (email/lead exhaust) filtered
   out.
2. **Semantic vector recall**: cosine search over the unified vector index,
   deduped against anything the FTS block already surfaced, thresholded at
   `PREFETCH_SIM_THRESHOLD` (0.40) so weak matches don't pollute the prompt.
3. **Graph neighborhood**: entities named in the query (or the rolling
   conversation window) get their 1-hop graph neighbors pulled in as
   `subject RELATION object` lines.
4. **Analogy candidate**: see below. At most one line.
5. **Situation match**: named recurring situations (see below). At most two.

A **rolling window** of the last N user messages (default 3) is used as the
query for vector/analogy/situation matching instead of just the current
turn, so multi-turn context doesn't get lost between messages.

**Chunk awareness**: if both a chunk fact and one of its summarized members
surface in the same prefetch, the member is suppressed (the chunk already
covers it), retrievable in full via the `chunk_expand` tool if needed.

Every surfaced fact gets its `retrieval_count` bumped and a row written to
`recall_log` (session, turn, query, which block, score); this is the raw
signal the nightly feedback pass consumes.

### 4. The analogy slot (Hofstadter-style reminding)

This is the most unusual mechanism in the system. See
[DESIGN-NOTES.md](DESIGN-NOTES.md) for the conceptual framing; this section
covers the mechanics.

Facts are encoded as HRR (Holographic Reduced Representation) phase vectors:
`bind(encode_text(content), role_content)`, content bound to a role via
elementwise phase addition. Binding is invertible under cosine similarity in
a specific way: `sim(bind(q, r), bind(c, r)) == sim(q, c)`. That means a
query can be probed against the *same role binding* every fact was written
with, and get back pure content similarity, but computed through the HRR
superposition, which behaves differently from a standard embedding: pairwise
similarities compress into a tight band around ~0.5 (superposition noise)
rather than spreading out the way cosine similarity over a dense embedding
does.

Because of that compression, the gate is **relative, not absolute**: a
candidate must stand `analogy_z_min` (default 2.0) standard deviations above
the mean similarity across the whole fact store, not just clear a fixed
threshold. On top of that, a **surface-similarity ceiling**
(`analogy_surface_max`, default 0.30) is checked using a *different*
embedding model (MiniLM, the same one behind normal vector recall), so a
candidate has to be structurally close under HRR while being superficially
*far* under a conventional embedding. That combination is what makes the
result feel like "this reminds me of..." rather than "this is similar to...".
At most one candidate per turn, never repeating anything shown recently
(tracked via `recall_log` where `block='analogy'`), injected as a clearly
labeled "speculative reminding" the agent is told to treat as a hint, not a
fact about the current topic.

The same fingerprinting idea, applied to the *graph* instead of fact content,
powers the `analogize` tool: two entities are analogous if their relational
fingerprints (multiset of `(direction, relation, neighbor_type)` triples) are
cosine-similar while their names are surface-different (guarded by a
`difflib` ratio ceiling so near-identical names don't count as their own
analogy).

### 5. Chunking and category promotion (weekly)

`memory_abstract.py` runs two independent, both-reversible mechanisms:

**Chunking.** Union-find clusters facts that share ≥2 entities, or share one
entity that isn't a "hub" (linked to more than `HUB_MAX_FACTS`=15 other
facts; without this guard, the owner's own name would transitively connect
nearly the entire fact store into one giant cluster). Components of 3-12
facts, all members ≥7 days old, get summarized by an LLM into one `category='chunk'`
fact. Members are **never deleted**. `chunk_members` rows record the
mapping, the hybrid provider suppresses member lines when their chunk
surfaces in prefetch, and `chunk_expand` unpacks them on demand. This is
lossy-looking but lossless: detail compresses out of default view without
being destroyed.

**INSTANCE_OF promotion.** Separately, graph nodes of the same type whose
relational fingerprints are near-identical (cosine ≥0.8) get grouped, and an
LLM names the emergent category (e.g. a cluster of company nodes that all
have `PROSPECT` + `LOCATED_IN <same metro>` edges might get named
`"regional prospects"`). The proposal is written as `INSTANCE_OF` edges
tagged `src_tag='proposed'`, and, until a human approves it, it is **hidden
from every reader** (prefetch and all graph tools), so an unreviewed guess can
never surface as if it were ground truth. Approval, rejection, and the review
queue are handled by the `ontology_review` tool; see
[Operator review of graph proposals](#13-operator-review-of-graph-proposals).

### 6. Trust feedback (nightly)

`memory_feedback.py` closes the loop on `recall_log`. A surfaced fact counts
as *engaged* if any later query in the same session either names one of its
linked entities or overlaps its content tokens (Jaccard ≥0.3), the cheapest
honest proxy for "the user came back to this topic after seeing it." Engaged
facts get `+0.02` trust (ceiling 0.85); facts surfaced on ≥5 distinct turns
in a 14-day window with *zero* engagement anywhere in that window get `-0.02`
(floor 0.35, deliberately above the `min_trust_threshold` of 0.3, so
demoted facts rank lower but never vanish from recall entirely). Every change
is logged to `trust_log` for a full audit/rollback trail.

### 7. Consolidation (weekly, quality, never age)

`memory_consolidate.py` operates only on the auto-extracted fact layer (never
the curated `MEMORY.md`/`USER.md`), and explicitly does **not** take a fact's
age as an input, ever. Two operations:

- Lossless dedup: identical-content facts collapse to the single best copy
  (most helpful → most retrieved → highest trust → oldest id, in that order),
  ties broken toward the oldest so the earliest-established version survives.
- Junk archival: facts matching narrow, explicit raw-transcript markers
  ("Voice call just ended", direction headers) get archived, but only
  if never protected (never recalled, never rated helpful, never trust-boosted
  above default). A separate `DURABLE` pattern (phone numbers, emails, dates,
  preference language) exists as a guard for future, broader junk heuristics
  but is deliberately *not* applied to the transcript path, since raw
  transcripts routinely contain incidental numbers that would otherwise look
  "durable."

Everything removed is appended to `memory_archive.jsonl` (full row, with
reason) before deletion; nothing is destroyed without a recoverable copy.

### 8. Calendar → graph episodes

`calendar_graph_sync.py` treats the personal calendar as a second ingest
source, distinct from the LLM-distillation path: every upcoming event becomes
an `event`-type graph node (`<summary> (<start-date>)`, idempotent on the
calendar API's `eventId`) wired with `ATTENDS` (from the owner node),
`INVOLVES` (any known person/company node whose name appears in the event
text/location), and `LOCATED_IN` (any known location node matched the same
way). This makes trips, meetings, and deadlines relationally queryable
(`graph_query('Prague')`, or via `analogize`) with real ISO dates attached,
instead of living only as prose facts an LLM had to notice and extract.
Future events that get cancelled or renamed are pruned automatically (past
episodes are left alone; they're history now); all calendar-sourced edges
carry `src_tag='calendar'` so they can be wiped and rebuilt independently of
everything else.

### 9. Folding structured data into the graph (pattern, not included)

The same primitives generalize to any structured, non-text data source: read
records, canonicalize names through `entity_resolve.Resolver` (passing
whatever strong identity keys the source has (domain, email, phone) so
records collapse onto the same nodes as email/call/calendar mentions of the
same entity), emit one graph node + a hub edge + one concise summary fact per
record, and hand the batch to `memory_ingest._append_overlay()`. A production
instance of this pattern imports a structured CRM/leads dataset this way; it
isn't included in this snapshot since the classification logic is
dataset-specific, but the shape is exactly `entity_resolve` +
`_append_overlay` + one `add_fact` call per record; see
`world_model_import.py`'s role in the ingest cron for where it plugs in.

### 10. Introspection

`brain_viz.py` is a read-only mirror: it loads the graph, computes hub
degrees and relationship-type histograms, and writes a self-contained
interactive HTML explorer (vis.js, search, click-to-inspect, type filters,
a "hide prospects/leaves" toggle) plus a Mermaid diagram of the highest-degree
core entities and a handful of PII-filtered sample facts, all into one
Markdown file that renders directly on GitHub. Nothing it writes is fed back
in. It exists purely so the memory is inspectable instead of a black box.

### 11. Provenance

Every fact and every edge carries a `source_ref` column stamped at write time,
so a surfaced memory is traceable back to what produced it. `memory_ingest.py`
stamps `<source>:<docid>` (first-assertion semantics: the ref keyed on the
`fact_id` the store returns, so the earliest document to assert a fact owns
it); `calendar_graph_sync.py` stamps `gcal:<eventId>`. Consumed ingest spools
are archived (not deleted) under `ingest/archive/` so those refs stay
dereferenceable. The `recall` tool and `graph_query` neighbor listings surface
the ref as a `src` field. One honest gap: the batch edge-mining pass
(`extract_edges_for_new_facts`) infers edges across many facts at once and
can't attribute a single source, so those edges keep an empty ref by design.
The `edges.source_ref` column is added idempotently by `memstore._ensure_column`
in `init_schema`; the fact-side column (the `facts` table is owned upstream) is
added once by `migrations/2026-07-provenance.py`, so an existing
`memory_store.db` converges without a rebuild.

### 12. Decision traces

`decision_log` (a table in `memstore`, plus the `decision_log` tool on the
provider) records what the agent *recommended* versus what the user actually
*did*. On `action='record'` it snapshots the recommendation and auto-links the
`recall_log` rows from that same turn as `source_refs`, i.e. the facts the
recommendation was (potentially) based on. On `action='resolve'` the outcome
(`accepted` / `modified` / `rejected`) is written back, falling back to the
latest pending decision in the session (or, if the user replies much later,
the latest pending anywhere). The nightly feedback pass then turns outcomes
into signal: `memory_feedback.sweep_decisions` ages any decision left pending
more than 7 days to `ignored`, and applies a small trust bump (`+0.02`,
exactly once, guarded by a `trust_applied` flag) to the facts linked under an
`accepted` or `modified` decision, on the theory that memory which fed a
recommendation the user took is memory worth trusting a little more. This is
deliberately the same proposal -> outcome -> reinforcement shape as the trust
loop in section 6, applied one level up, at the recommendation rather than the
individual fact.

### 13. Operator review of graph proposals

Two of the mechanisms above generate *proposals*, not facts: fuzzy
entity-resolution near-misses (`POSSIBLE_ALIAS`, `src_tag='alias-candidate'`,
section 2) and emergent-category groupings (`INSTANCE_OF`, `src_tag='proposed'`,
section 5). Neither is trusted until a human says so. `ontology_review.py`
(library + CLI, and the `ontology_review` tool on the provider) is the gate:

- `list` shows the pending queue (alias pairs and proposed categories).
- `approve alias` calls `Resolver.add_alias` (so *future* writes canonicalize)
  and `memstore.merge_nodes` (so the *existing* graph collapses the two nodes).
- `reject alias` / `reject category` flips the edge to `src_tag='rejected'`
  rather than deleting it: the `(src, rel, dst)` primary key then makes any
  future re-proposal an `INSERT OR IGNORE` no-op, so a rejection is durable and
  the machine doesn't keep re-suggesting the same bad merge.
- `approve category` flips the `INSTANCE_OF` edges to `src_tag='curated'`.

The enforcement point is `memstore.load_graph`, which excludes
`proposed`, `rejected`, and `alias-candidate` tags by default, so unreviewed
(and rejected) proposals are invisible to prefetch and every graph tool until
approval promotes them. A weekly cron (`ontology_review.py --nudge`) prints the
queue only when it's non-empty, and stays silent otherwise, so the human is
pinged to review exactly when there's something to review. This is the
falsifiability tenet from [DESIGN-NOTES.md](DESIGN-NOTES.md) made operational:
the system may *propose* structure, but a person confirms it before anything
depends on it.

### 14. Entity-type sync

Facts and the graph learn entity types on different paths (the graph gets a
`type` from the ingest extractor; the fact-side `entities` table often
doesn't), so `memstore_sync.sync_entity_types` copies `graph_nodes.type` onto
any untyped `entities.entity_type`, canonicalizing names through
`entity_resolve` so the two sides line up. It rides the 15-minute ingest cron
and is idempotent (only ever fills blanks). This is what lets same-type
constraints (analogy candidate typing, category fingerprinting) work on the
fact side, not just the graph side.

## Config reference

```yaml
memory:
  provider: hybrid

plugins:
  hybrid:
    prefetch_char_cap: 2000       # hard cap on the injected block (0 = off)
    count_retrievals: true        # bump facts.retrieval_count on surfacing
    recall_log_enabled: true      # write recall_log rows for the nightly pass
    operational_categories: [email, lead]        # excluded from default prefetch
    operational_vec_types: [email, lead, job_application]
    rolling_window: 3             # user messages in the situation window
    analogy_slot: true
    analogy_hrr_min: 0.50         # absolute HRR sim floor
    analogy_z_min: 2.0            # std-devs above the store's similarity mean
    analogy_surface_max: 0.30     # MiniLM cosine ceiling (surface-different)
    analogy_min_query_chars: 40
    situation_sim_min: 0.42
    analogize_enabled: true

  # Owned by the upstream holographic provider, not this plugin, but read
  # directly by memory_consolidate.py (is_protected()) and memory_abstract.py,
  # so it belongs in this reference too.
  hermes-memory-store:
    auto_extract: true
    default_trust: 0.5              # baseline trust_score for new facts
    min_trust_threshold: 0.3        # floor below which facts stop surfacing
    temporal_decay_half_life: 0     # 0 = disabled; this system ages trust by
                                     # engagement (memory_feedback.py), not a
                                     # built-in clock

cron:
  memory_enabled: true              # opt cron-triggered agent runs into
                                     # memory read/write (see patches/)
```

Everything degrades gracefully: if the vector store, graph, numpy/HRR, or the
upstream holographic base is unavailable, the provider falls back to
whichever layers still work: worst case, exactly the bundled holographic
behavior with none of the above.

## Testing

`scripts/tests/` is a real, currently-green pytest suite (59 tests) covering
the pure-function and schema-level behavior of every mechanism above:
`memstore` schema/vector/graph primitives, `entity_resolve`'s merge/fuzzy/
alias-candidate logic and the closed relation vocabulary (`canon_relation`
exact/flip/stem resolution), the one-time relation-remap migration,
`memory_abstract`'s cluster rules (hub guard, size bounds, age gate,
never-delete invariant), `memory_feedback`'s engagement heuristic, trust
clamps, and decision-outcome sweep, `calendar_graph_sync`'s event-naming and
staleness rules, `ontology_review`'s approve/reject/durable-rejection logic,
the `decision_log` record/resolve flow, and the hybrid plugin's
entity-candidate hygiene and pure helper functions (`_cap_blocks`,
`_content_words`, `_fact_id_from_vid`, `_counter_cosine`). None of it touches a
real LLM or network call; `facts_db` and `mem_db` fixtures in `conftest.py`
build a throwaway schema per test.

`conftest.py` documents its own purpose plainly: it doubles as a smoke suite
against upstream breakage: if a Hermes Agent update shifts the holographic
base class's shape, these tests catch it before the live gateway does.

```
HERMES_HOME=<scratch dir> PYTHONPATH=scripts:<path-to-hermes-agent> \
  <hermes-agent venv>/bin/python3 -m pytest scripts/tests/ -q
```

(`PYTHONPATH` needs both `scripts/` (for `memstore`, `entity_resolve`, etc.)
and a `hermes-agent` checkout, since a handful of the test files load
`plugins/hybrid/__init__.py` directly and it imports the upstream holographic
base class.)

## Plugging into Hermes Agent

Hermes Agent defines a `MemoryProvider` ABC (`agent/memory_provider.py`) with
a documented plugin contract (drop a `plugins/<name>/{__init__.py,plugin.yaml}`
into `$HERMES_HOME/plugins/`, implement `name`, `is_available()`,
`initialize()`, `get_tool_schemas()`, `handle_tool_call()`, plus optional
hooks like `prefetch()`, `system_prompt_block()`, `on_memory_write()`).
`HybridMemoryProvider` subclasses the *bundled* holographic provider rather
than the bare ABC, specifically so plugin installs stay update-safe: if the
bundled provider ever moves or changes shape, the import falls back to the
bare ABC and the plugin degrades to vector+graph-only instead of breaking.
This system depends on one small local patch outside the plugin architecture
itself: `patches/cron-memory-opt-in.diff`, against `cron/scheduler.py`. Cron
(scheduled, non-interactive) agent runs hardcode `skip_memory=True` upstream
(reasonably, since a cron system prompt shouldn't get treated like a normal
conversational turn by default), but that also means cron-triggered agents
can't read or write memory at all. The patch gates it behind an opt-in
`cron: memory_enabled: true` config flag instead of a hardcoded `True`, so
cron jobs can participate in memory when that's actually wanted. It's a
single file, so it's easy to reapply (a diff comment even says so) after an
upstream update overwrites it, but it's real fork-maintenance burden, and
the honest accounting of "what does this cost to keep running" should include
it, not just the update-safe plugin.

## History

An earlier prototype (`plugins/semantic/`, not included here) explored
similar territory (entity extraction, temporal/lifetime classification,
pruning heuristics) as a set of standalone agent tools rather than a real
`MemoryProvider`. It's disabled in the live config now, superseded by the
architecture in this repo; it's worth naming mainly as a data point that this
wasn't a one-shot design: the unified-store, entity-resolution, and
graph-first version is a second (and materially different) iteration.

## Known limitations

- Trust-feedback engagement detection is a heuristic (token-Jaccard + entity
  mention), not LLM-judged; `--llm` is a reserved, unimplemented flag for a
  v2 pass.
- Single-tenant, single-SQLite-file. No sharding story; fine at thousands of
  facts/entities, untested well beyond that.
- The analogy slot's quality is bounded by the HRR encoding's simplicity
  (bag-of-words style content encoding, no syntax/order sensitivity). It
  finds analogies in *what entities and relations are present*, not in
  deeper structural or causal patterns.
- Category promotion and chunk summarization both depend on an LLM call
  succeeding; failures just skip that cluster until the next weekly run (no
  retry queue).
