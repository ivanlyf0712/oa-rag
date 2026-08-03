# 03 — Graph rewrite on real conversation relationships

**What to build:** The graph layer is rebuilt on genuine conversation structure — sender, receiver, chatroom (`open_kfid`), company — instead of vector similarity. A user searching with graph expansion enabled gets results that surface real connections (e.g. who talks to whom about what), not a similarity echo. The base regression suite stays green.

**Blocked by:** 01 — Chinese-capable hybrid base

**Status:** resolved

- [x] Graph nodes/edges encode real relationships: sender, receiver, chatroom, company, label
- [x] Graph expansion surfaces genuine connections rather than vector-similarity neighbors
- [x] Graph expansion verified to improve (or at worst match) base-only retrieval on the regression suite
- [x] Regression suite stays green with graph expansion on

## Notes

This is the design-heavy ticket. It may need its own grilling/spec session before implementation — the current graph is built on vector similarity, and the redesign to real conversation edges is a design task, not a bug fix.

## Answer

**Design:** The graph was rebuilt on real conversation structure. One node per message chunk; five structural edge types computed deterministically from chunk metadata at index time (`same_conversation`, `sender_receiver`, `same_sender`, `same_company`, `same_label`). Only the first four are traversal-eligible; `same_label` is recorded but never traversed so a label never acts as a match signal. Because every node is guaranteed at least one structural edge (the `same_label` edges), txtai's `approximate: true` graph inference auto-skips all nodes — the graph is purely structural with no vector inference, no LLM calls, and no randomness.

**Expansion:** `_graph_expand` was rewritten to use the direct backend API (`graph.edges(node_key)`) instead of Cypher (GrandCypher is unavailable in this env). It walks the 4 traversal-eligible edge types from the top-3 base results, one hop, and appends neighbor documents below the base ranking. Each expanded document is scored as `parent_score × 0.8 × neighbor_query_relevance` (the query-consistency gate), so a structurally-connected but query-irrelevant neighbor is balanced out instead of blindly followed. Expansion is append-only: base order/membership is never reordered or displaced.

**What changed:** `_compute_structural_relationships` helper added to `search.py`; `IndexBuilder._fetch_messages` SQL + chunk metadata now carry `company`; `IndexBuilder.build` passes dict docs with `relationships` field + `columns.relationships` config; `_graph_expand` rewritten with direct backend API, query-consistency gate, and append-only semantics; `search()` passes `output[:limit]` to `_graph_expand` and no longer double-truncates.

**Verification:** 17/17 tests pass (7 new graph tests + 4 ticket-01 regression + 6 ticket-02 expansion). Graph expansion keeps base relevance (logistics `物流`+`報價` and investment `債券`+`藍籌` messages still in top-3), surfaces genuine same-conversation connections tagged with `_graph_relation`, protects labels under expansion (bare-label search doesn't rank all docs of that label), and the query-consistency gate ensures irrelevant neighbors don't surface.

**Context pointer:** See `docs/adr/0001-structural-conversation-graph.md` for the architectural decision record, `CONTEXT.md` for the domain glossary, and `.scratch/retrieval-base-consolidation/graph-rewrite-spec.md` + `.scratch/retrieval-base-consolidation/graph-rewrite/` for the spec and tracer-bullet tickets.