# Design notes: why "Hofstadter-style"

Most agent memory systems treat retrieval as a nearest-neighbor problem:
embed the query, embed the store, return what's closest. That's a reasonable
default and this system uses it too (the vector recall block). But it isn't
the whole story of how recall actually seems to work, and two ideas from
cognitive science motivated the parts of this system that go further:

## Analogy as the core mechanism, not a special case

Douglas Hofstadter's central, decades-long argument (from *Gödel, Escher,
Bach* through *Fluid Concepts and Creative Analogies* and *Surfaces and
Essences*) is that analogy-making isn't a rare, fancy cognitive trick —
it's the basic mechanism by which categories form and recall happens at all.
Recognizing that a new situation is "like" an old one, despite being
superficially different, is what lets one experience generalize to the next.

The **analogy slot** in `prefetch()` is a literal, if simple, attempt at
that: instead of only asking "what's most similar to this query," it asks
"what's structurally similar while being superficially *different*" — using
one embedding space (HRR phase vectors, encoding content abstractly through
binding) to find structural closeness, and a second, unrelated embedding
space (MiniLM) as a surface-difference filter. The two spaces disagreeing is
the signal: high structural similarity, low surface similarity, is what a
human "oh, this reminds me of..." moment actually looks like, as opposed to
"this is the same topic."

This is a narrow, mechanical instance of the idea — it operates on
bag-of-entities/content signatures, not the rich perceptual/conceptual
representations Hofstadter's own Copycat architecture worked with. It doesn't
claim to be a general analogy engine. But it's a genuine attempt to make
*structural* reminding a first-class retrieval mode instead of collapsing
everything to "how similar is this text."

## Categories are made, not given

The second idea, mainly from *Surfaces and Essences*: categories aren't
fixed containers things get sorted into — they're actively, continually
constructed from noticing that several things play the same role. A "chair"
is whatever affords sitting; the category is discovered through use, not
handed down in advance.

Two mechanisms here try to take that seriously instead of treating the graph
schema as fixed:

- **Chunking** notices that a cluster of facts keeps co-occurring around
  shared entities and reifies that cluster into a new thing — a chunk fact,
  with its own graph `episode` node — without losing the ability to unpack it
  back into the originals. The chunk *is* a new category member, not a
  cache.
- **INSTANCE_OF promotion** looks for graph nodes that, despite having
  different names, play structurally identical roles (same relation types to
  the same kinds of neighbors) and proposes that they're instances of an
  emergent category — one the system didn't start with, named by an LLM
  after the fact, reviewable before anything trusts it.

Neither mechanism is allowed to happen silently and unreviewably: chunks keep
their members, category proposals are tagged `src_tag='proposed'` and
wipeable in one call. The goal is categories that *emerge from use* while
staying falsifiable — closer to how Hofstadter describes concepts actually
forming than to a fixed ontology decided in advance.

## What this explicitly doesn't claim

This isn't a reimplementation of Copycat, isn't doing genuine structure-
mapping over rich relational representations, and the "analogy" it finds is
bounded by how much signal a bag-of-entities HRR encoding can carry. The
honest description is: two cheap, literal mechanisms inspired by a couple of
Hofstadter's central claims, wired into an otherwise fairly conventional
facts+vectors+graph memory stack, because those claims seemed worth taking
seriously as *retrieval design constraints* rather than leaving them as
philosophy.
