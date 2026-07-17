# Hybrid Memory: a graph + vector + holographic memory layer for agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Plugin for hermes-agent](https://img.shields.io/badge/plugin%20for-hermes--agent-6f42c1.svg)](https://github.com/NousResearch/hermes-agent)
[![tests](https://github.com/sheldon904/hermes-hybrid-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/sheldon904/hermes-hybrid-memory/actions/workflows/tests.yml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> Facts + semantic vectors + a knowledge graph in **one SQLite file**, maintained
> incrementally as a side effect of normal memory writes, plus a Hofstadter-inspired
> analogy slot, emergent chunking, and usage-driven (never age-driven) forgetting.

[![The memory graph rendered by brain_viz.py](docs/img/brain_graph.png)](docs/EXAMPLE-SNAPSHOT.md)

<sub>The memory isn't a black box: `brain_viz.py` snapshots the whole store into a
self-contained interactive explorer, where you can search, click any entity to see its links and
what the agent knows about it, filter by type, and hide the prospect leaves. *(Shown on
synthetic demo data; [browse the full snapshot](docs/EXAMPLE-SNAPSHOT.md), Mermaid map included.)*</sub>

A custom `MemoryProvider` plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)
(MIT-licensed, NousResearch) that replaces "stuff some text in the system prompt"
memory with a unified store combining **structured facts**, **semantic vector
recall**, and a **knowledge graph**, plus a set of Hofstadter-inspired
mechanisms (analogy-making, chunking, emergent categories) layered on top.

For an independent, data-backed look at how this behaves in production, see
[**Hybrid Holographic Memory in a Production Personal Agent**](docs/hermes-hybrid-memory-study.pdf),
a seventeen-day empirical characterization covering store composition and
growth, recall-channel telemetry (1,976 events / 235 sessions), a 200-query
known-item retrieval probe, a full replay of every analogy-slot firing, and
latency microbenchmarks, measured read-only against a live instance.

This repo is a staged extraction of a system that has been running in
production against a real personal/work agent for several weeks. Some paths
still assume the `$HERMES_HOME` layout it grew up in; see [Status](#status)
below for what that means if you want to run it elsewhere.

## Why

Most agent "memory" is one of two things: a system-prompt scratchpad file, or
a vector store you dump text into and cosine-search back out. Both work until
the agent needs to reason about *how things relate* ("which of these leads
looks like our best client," "how do X and Y connect," "have I been in this
situation before"), at which point flat text recall stops being enough.

The bet here: keep the fast, cheap paths (keyword/FTS recall, vector
similarity) but add a real graph underneath, maintained incrementally and for
free as a side effect of normal memory writes, and make retrieval intentional
about a case cognitive-science research keeps landing on: that useful recall
isn't just "most similar," it's sometimes "structurally similar but
superficially different" (analogy), and long-term memory keeps compressing
raw experience into reusable categories (chunking, concept formation).

## Architecture at a glance

```
                         ┌─────────────────────────────┐
   documents in ────────▶│   memory_ingest.py           │
   (email, calls,        │   LLM distills facts +       │──▶ entity_resolve.py
    notes, chat)         │   graph nodes/edges (JSONL)   │    (canonicalize names)
                         └─────────────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │      memory_store.db (SQLite)          │
                     │  ┌───────────┐ ┌──────────┐ ┌────────┐│
                     │  │ facts +   │ │ sqlite-vec│ │ graph  ││
                     │  │ FTS5 +    │ │ vectors   │ │ (nodes/││
                     │  │ trust     │ │ (384-dim) │ │ edges) ││
                     │  │(upstream) │ │(memstore) │ │(memstore)│
                     │  └───────────┘ └──────────┘ └────────┘│
                     └──────────────────────────────────────┘
                          ▲            ▲              ▲
              memstore_sync.py   memory_abstract.py   calendar_graph_sync.py
              (incremental        (weekly: chunking +   (mirrors calendar
               vectorize + gist    category promotion)    events as graph
               + edge-mine)                                episodes)
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │   HybridMemoryProvider (plugins/hybrid) │
                     │   every turn: prefetch() assembles     │
                     │   FTS + vector + graph + analogy +     │
                     │   situation blocks, char-capped        │
                     └──────────────────────────────────────┘
                                        │
                                        ▼
                              injected into the agent's
                                 system context, every turn
```

Nightly and weekly passes close the loop: `memory_feedback.py` adjusts trust
scores from what actually got engaged with, and `memory_consolidate.py` dedupes
and archives noise (never by age, only by demonstrated uselessness).

## Install

This is a plugin for [hermes-agent](https://github.com/NousResearch/hermes-agent),
so it installs into an existing `$HERMES_HOME`:

```bash
# 1. dependencies (into the hermes-agent venv)
pip install -r requirements.txt

# 2. drop the plugin in place
cp -r plugins/hybrid   "$HERMES_HOME/plugins/hybrid"
cp    scripts/*.py     "$HERMES_HOME/scripts/"

# 3. activate it in $HERMES_HOME/config.yaml
#    memory:
#      provider: hybrid
#    (see docs/ARCHITECTURE.md#config-reference for the full block)

# 4. (optional) wire the cron passes (ingest / consolidate / feedback)
#    and apply patches/cron-memory-opt-in.diff so cron runs can touch memory
```

An `OPENROUTER_API_KEY` (env var or `$HERMES_HOME/.env`) enables the LLM-backed
steps (ingest distillation, gisting, chunk/category naming, graph reasoning);
everything else runs without it. See [Status](#status) for the caveats of running
this outside the `$HERMES_HOME` layout it grew up in.

## What's actually novel here

- **One unified store.** Facts, vectors, and graph edges all live in one
  SQLite database (`memory_store.db`), written incrementally. No batch
  rebuild step, no separate vector DB to keep in sync (a prior ChromaDB-based
  version was retired for exactly that reason).
- **Write-time entity resolution.** Every name (from email senders, call
  transcripts, calendar events, structured business data) resolves through a
  canonicalization layer (`entity_resolve.py`) before it becomes a graph node,
  so "Acme Labs," "acme-labs.com," and an email from `@acme-labs.com` all
  collapse to one node, while genuinely distinct near-namesakes stay separate.
- **An analogy slot, for real.** Every turn, one memory can surface not
  because it's *similar* to the current conversation but because it's
  *structurally* similar while being *superficially different*, using
  Holographic Reduced Representations (HRR / phase vectors) to compute
  content similarity independent of surface wording, gated against a
  surface-similarity ceiling from a separate embedding model. This is a
  direct, literal implementation of Hofstadter's argument that analogy is the
  core of cognition, not a side quest. See
  [docs/DESIGN-NOTES.md](docs/DESIGN-NOTES.md).
- **Chunking and emergent categories.** A weekly pass clusters related facts
  (union-find over shared entities, with a "hub guard" so the owner's own
  name doesn't transitively connect everything) and asks an LLM to compress
  each cluster into one summary fact, without ever deleting the originals.
  Separately, graph nodes with near-identical relational fingerprints get
  proposed as members of an emergent category node, reviewable before it's
  trusted.
- **Usage is the feedback signal, not age.** Nothing is ever pruned for being
  old. A fact's trust score moves based on whether it actually gets
  *engaged with* after being surfaced (the user's next query touches its
  entities or content). Recalled-but-ignored facts quietly rank lower;
  recalled-and-used facts get reinforced.
- **Provenance on every write.** Each fact and edge carries a `source_ref`
  stamped at write time (`<source>:<docid>`, `gcal:<eventId>`), and consumed
  ingest spools are archived rather than deleted, so a surfaced memory can be
  traced back to the email, call, or calendar event that produced it. `recall`
  and `graph_query` surface it as a `src` field.
- **Machine proposes, human disposes.** The two mechanisms that *guess* (fuzzy
  entity-alias merges and emergent-category groupings) never surface as truth:
  their edges are tagged (`alias-candidate` / `proposed`) and hidden from
  prefetch and every graph tool by default until a human approves them through
  the `ontology_review` tool. Rejections are durable (the edge is kept and
  tagged `rejected` so the same bad merge can't be re-proposed), and a weekly
  cron nudges for review only when the queue is non-empty.
- **Decision traces close a second loop.** A `decision_log` tool records what
  the agent recommended versus what the user actually did, auto-linking the
  facts surfaced that turn; the nightly pass then nudges trust on the memory
  behind recommendations the user took. Same proposal -> outcome ->
  reinforcement shape as the fact-level trust loop, one level up.
- **A live graph browser.** `brain_viz.py` snapshots the whole store into a
  self-contained interactive HTML graph (vis.js) plus a Mermaid map of the
  core, so the memory is inspectable, not a black box.

## Operational cadence

Everything runs as cron jobs against a live `HERMES_HOME`, not inside the
request path (except `prefetch()`, which is deliberately fast: SQLite reads
and cached lookups only, no LLM calls on the hot path):

| Schedule | Job | Does |
|---|---|---|
| every 15 min | `memory-ingest` | drain spooled emails/calls → `memory_ingest.py` → `calendar_graph_sync.py` → `memstore_sync.py` (vectorize + gist + edge-mine + `sync_entity_types`) |
| weekly (Mon 04:00) | `memory-consolidate` | `memory_consolidate.py --apply` (dedup + junk archival) → `memory_abstract.py --apply` (chunking + category proposals) → resync |
| weekly (Mon 09:00) | `ontology-review-nudge` | `ontology_review.py --nudge` (message the pending alias/category queue for review; silent when empty) |
| nightly (03:45) | `memory-feedback` | `memory_feedback.py --apply` (trust adjustment from engagement, decision-outcome sweep, recall-log pruning) |

As of this writing the live deployment's graph has grown to roughly 1,000
nodes and 1,600 relationships purely from normal agent use (chat, ingested
email/calls, calendar); nobody hand-curates the graph, machine-proposed merges
and categories only enter it after the human approves them.

## Repo layout

```
plugins/hybrid/          the MemoryProvider plugin itself (drop-in for
                          $HERMES_HOME/plugins/hybrid/)
scripts/                 the standalone pipeline scripts (run via cron,
                          independent of any single agent session)
  memstore.py               unified store: schema + vector/graph primitives
  entity_resolve.py         write-time name canonicalization
  memory_ingest.py          document -> facts + graph (LLM distillation)
  memstore_sync.py          incremental vectorize + gist + edge mining
  memory_abstract.py        weekly: chunking + emergent category promotion
  memory_consolidate.py     dedup + junk archival (never by age)
  memory_feedback.py        nightly: usage-based trust + decision-outcome sweep
  calendar_graph_sync.py    mirrors calendar events into the graph as episodes
  ontology_review.py        operator gate for proposed aliases/categories
                             (library + CLI + weekly --nudge)
  brain_viz.py              snapshot the store as an interactive graph explorer
  backfill_hrr.py           one-shot: recompute HRR vectors for existing facts
  backfill_gists.py         one-shot: backfill gists for existing facts
  migrations/               idempotent one-time schema/data migrations
                             (provenance columns, relation-vocabulary remap)
  tests/                    pytest suite (schema, entity resolution + relation
                             vocab, clustering, feedback, decision sweep,
                             calendar sync, hybrid helpers)
docs/
  ARCHITECTURE.md           deep technical dive: data flow, config, internals
  DESIGN-NOTES.md           the Hofstadter framing: why analogy/chunking/categories
  hermes-hybrid-memory-study.pdf
                            17-day empirical characterization: telemetry, retrieval
                            benchmarks, analogy-slot replay, latency, on live data
patches/
  cron-memory-opt-in.diff   the one local patch against hermes-agent itself
                             (see Status below)
```

## Testing

```
PYTHONPATH=scripts:<path-to-a-hermes-agent-checkout> \
  python3 -m pytest scripts/tests/ -q
```

59 tests, no network/LLM calls, no real HERMES_HOME needed, with throwaway SQLite
fixtures per test. See `docs/ARCHITECTURE.md#testing` for what's covered and
why a few of the test files need an upstream `hermes-agent` checkout on the path.

## Status

This is a staged extraction, not yet a standalone package:

- Everything assumes a `$HERMES_HOME` directory layout (config.yaml, a
  `scripts/` dir on `sys.path`, a `hermes-agent` venv). Making this installable
  outside that context means factoring out `HERMES_HOME`-relative paths and
  the upstream `HolographicMemoryProvider` import into an explicit optional
  dependency.
- `plugins/hybrid/__init__.py` subclasses a bundled provider
  (`plugins.memory.holographic.HolographicMemoryProvider`) that ships with
  Hermes Agent itself. It is **not** included here (it's upstream, MIT
  licensed, see [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)).
  The subclass degrades gracefully to vector+graph-only if that import fails,
  which is what makes it update-safe against upstream changes.
- One business-specific pipeline (folding a structured CRM/leads dataset into
  the graph as a "world model," using the same canonicalization + overlay-write
  pattern as everything else) was intentionally left out of this snapshot
  since it's tied to a specific dataset; the pattern is described in
  `docs/ARCHITECTURE.md`.
- `patches/cron-memory-opt-in.diff` is a small but real fork-maintenance
  cost: it edits `cron/scheduler.py` inside Hermes Agent itself (not this
  plugin), and gets silently reverted by any upstream update that touches
  that file. Without it, cron-triggered agent runs can't read or write
  memory at all; see `docs/ARCHITECTURE.md` for why.

## Dependencies

`sqlite-vec`, `networkx`, `numpy`, `PyYAML`, and `chromadb` (only for its
bundled ONNX `all-MiniLM-L6-v2` embedding function; the ChromaDB *store*
itself is not used). An `OPENROUTER_API_KEY` env var (or `.env` entry) is
required for the LLM-backed steps (ingest distillation, gisting, chunk
summarization, category naming, graph reasoning); everything else degrades
gracefully without it.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). The two
invariants worth knowing before you touch anything: every layer must degrade
gracefully when another is missing, and no mechanism is ever allowed to destroy
memory to save space.

## License

[MIT](LICENSE). Built as a plugin for
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
(also MIT); the bundled holographic base provider it subclasses ships with, and
belongs to, that upstream project.
