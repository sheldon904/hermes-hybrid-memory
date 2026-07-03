# Contributing

Thanks for taking a look. This started as a personal system extracted from a
live [hermes-agent](https://github.com/NousResearch/hermes-agent) deployment, so
a few things about how to contribute are worth stating up front.

## Ground rules

- **Keep the degrade-gracefully invariant.** Every layer (vector, graph, HRR,
  the holographic base) is optional at runtime; if one is missing the provider
  falls back to whatever still works. New code should preserve that; don't make
  a hard dependency on a layer that today is best-effort.
- **Never delete memory to save space.** Chunking, consolidation, and feedback
  are all designed to be reversible (members retained, archives written, trust
  floored rather than dropped). Keep that property.
- **No network or real LLM calls in tests.** The suite uses throwaway SQLite
  fixtures. If a change needs an LLM, gate it and make the test cover the
  no-key path.

## Running the tests

```bash
HERMES_HOME=<scratch dir> \
PYTHONPATH=scripts:<path-to-a-hermes-agent-checkout> \
  python3 -m pytest scripts/tests/ -q
```

Two of the eight test files import `plugins/hybrid/__init__.py`, which imports
the upstream holographic base class; that's why a `hermes-agent` checkout is on
`PYTHONPATH`. The other six run against `scripts/` alone. See
[`docs/ARCHITECTURE.md#testing`](docs/ARCHITECTURE.md#testing).

## Before opening a PR

- Run the test suite and keep it green.
- If you touch prefetch, ingest, or the analogy/chunking mechanics, update
  `docs/ARCHITECTURE.md` in the same PR; the docs are the spec here.
- Open an issue first for anything that changes the storage schema or the
  plugin contract with hermes-agent.

Bug reports and design discussion are very welcome via GitHub Issues.
