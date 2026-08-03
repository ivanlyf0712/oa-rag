# Spec: Graph Rewrite on Real Conversation Relationships

## Problem Statement

The graph layer is currently built on vector similarity (txtai's auto-inferred graph). It is a similarity echo: connected documents are "similar-sounding" rather than genuinely related in conversation. When graph expansion is enabled, it surfaces vector-similarity neighbors that carry no real conversational meaning, so it cannot answer the actual intelligence question the graph is for — "who talks to whom about what."

## Solution

Rebuild the graph on real conversation structure. The graph becomes the **conversation graph**: one node per message chunk, with deterministic structural edges computed from the message metadata — sender, receiver, chatroom (`open_kfid`), company, and label. Graph expansion then walks these structural edges to surface genuine connections while never reordering or displacing the verified base retrieval.

## User Stories

1. As a user searching `物流報價 方案`, I want graph expansion to surface the full logistics-quotation thread (including same-conversation follow-ups that share few keywords), so I get the conversation context, not a single message.
2. As a user searching `投資美國債券跟藍籌股`, I want graph expansion to surface the rest of the investment-opportunity conversation, so the RAG answer is grounded in the whole dialogue.
3. As a user searching a bare label such as `product_inquiry`, I want graph expansion to still not rank all documents of that label, so a label never becomes a match signal.
4. As a user filtering by label, I want graph expansion to never leak documents of other labels into the filtered results, so label filtering stays authoritative.
5. As a search UI user, I want results that came from graph expansion to be annotated with the relationship that surfaced them, so I can see *why* they appeared.
6. As a user, I want graph expansion to only ever add relevant results below the base ranking — never reorder or push out the base's top hits.
7. As a developer, I want the graph to be built deterministically from structured metadata, so index rebuilds are reproducible and tests are stable.
8. As a developer, I want the graph to be purely structural (no vector inference, no LLM calls), so there is no hidden non-determinism or external dependency at index time.
9. As a developer, I want the existing ticket-01 and ticket-02 regression suites to stay green with graph expansion enabled, so later layers cannot silently regress the verified base.

## Implementation Decisions

1. **Conversation graph node model.** One node per message chunk (the existing document id). Nodes are homogeneous — there are no separate entity nodes for people, companies, or labels. This keeps the graph aligned with the `Searcher.search()` seam, whose results are documents.
2. **Structural edges (all recorded).** Five structural edge types are computed deterministically from chunk metadata at index time:
   - `same_conversation` — chunks of different turns in the same chatroom (`open_kfid`)
   - `sender_receiver` — chunk A's sender is chunk B's receiver (or vice versa) in the same chatroom
   - `same_sender` — chunks from the same `external_userid`
   - `same_company` — chunks whose senders belong to the same `company`
   - `same_label` — chunks with the same `label`
3. **Traversal eligibility.** Only four edge types are traversal-eligible: `same_conversation`, `sender_receiver`, `same_sender`, `same_company`. `same_label` is recorded in the graph but never traversed by expansion, so a label can never act as a match signal (protects the label regression suite).
4. **Purely structural graph construction.** Structural edges are supplied at index time via txtai's manual `relationships` field on the document object (config `columns.relationships`). Because txtai's graph inference defaults to `approximate: true`, any node that already has an edge is skipped by vector inference. Every chunk is therefore guaranteed at least one structural edge (the `same_label` edges), so vector auto-inference never runs — the graph is structural with no inference, no LLM calls, and no randomness.
5. **Nothing to tune away from the base.** Graph expansion is an enhancement on top of the verified base. The base retrieval path (`graph_expand=0`) is unchanged, and expansion only appends results below the base ranking.
6. **Expansion semantics (append-only, non-tampering).** Graph expansion runs after base retrieval (hybrid, or RRF fusion when expansion is enabled) and after filtering. It walks the 4 traversal-eligible edges from each of the top-3 seed results, one hop, and appends newly-found neighbor documents that were not already in the result set. Expanded documents are re-checked against the active label/date filters. The base result list's order and membership are never modified — expanded documents are appended below it, each scored `parent_score × 0.8`, and the final list is truncated to the requested limit.
7. **Expanded results carry provenance.** Each expanded document is annotated with `_graph_relation` (the edge type that surfaced it) and `_from_node` (the seed it came from), so the UI/CLI can explain why the document appeared.
8. **All expansion knobs are parameters.** `max_expand` seeds (default 3), `hop_discount` (default 0.8), and the set of traversal-eligible edge types are `Searcher._graph_expand` parameters, so the behavior can be tuned later without reshaping the seam.
9. **Graph lifecycle.** The graph is enabled at index build time (`IndexBuilder.build`) and is purely structural. The old `graph_mode=llm` path (LLM relation extraction) is removed: it was non-deterministic, called a non-existent txtai API (`add_edge`), and is the text-guessing anti-pattern this ticket removes. `graph_mode=auto|llm` both resolve to the same structural construction; `graph_mode=off` keeps the graph disabled.
10. **Company metadata completeness.** The message fetch and chunk metadata must carry `company` through to graph construction (currently the chunk metadata lacks it), so `same_company` edges are computed from real data, not empty strings.
11. **Filtering correctness.** Expanded documents must pass the same label/date filter as base results. This closes a gap where graph expansion previously appended neighbors without re-checking filters.

## Testing Decisions

- A good test asserts **external behavior** through `Searcher.search()` — what a user sees — not implementation details of the index or the graph backend.
- The graph tests build a **deterministic in-memory index** from the conversation templates (same pattern as ticket 01/02 tests), but with graph enabled and structural relationships supplied. The test fixture is the only way the graph is exercised; the same seam is used for base-only and graph-expanded comparisons.
- The graph test module asserts:
  1. **Structural-only invariant:** every graph node has at least one structural edge; no vector-similarity edges are present, so the graph is purely structural.
  2. **Expansion keeps base relevant:** `graph_expand=1` top-3 still contains the logistics (`物流`+`報價`) and investment (`債券`+`藍籌`) messages, matching `graph_expand=0`.
  3. **Genuine connection surfaces:** `graph_expand=1` returns a same-conversation message that base-only top-3 misses, annotated with `_graph_relation` — e.g. the 報價單 follow-up for `物流報價 方案`.
  4. **Label protection under expansion:** a bare-label search with `graph_expand=1` does not rank all documents of that label; label-filtered search with `graph_expand=1` contains only that label.
  5. **Non-tampering / improve-or-match bar:** the top-3 base documents are present and in the same relative order with graph expansion on; graph expansion adds a relevant same-conversation document for at least one query.
- **Prior art:** the ticket-01 regression suite (tests asserting base relevance through `Searcher.search()`) and ticket-02 expansion suite (comparing `graph_expand=0` base against an enhancement layer). The graph suite extends the same pattern by toggling `graph_expand=1`.
- The existing `test_search_regression.py` and `test_search_expansion.py` files are **not modified**; they remain the permanent gate and run with `graph_expand=0`.

## Out of Scope

- Reranker work — ticket 04 (multilingual reranker) is a separate layer.
- Any change to the ticket-01 or ticket-02 regression assertions.
- Weakening base retrieval: the graph is an enhancement on top of the verified base.
- Entity-node graphs (optional future capability for entity-centric graph queries).
- LLM-based or vector-based relation inference at index time.

## Further Notes

- The corpus is ~100% Traditional Chinese; the production index is built on the machine-specific path pattern already in place. The structural graph adds no new external dependencies beyond txtai's networkx graph backend (already installed).