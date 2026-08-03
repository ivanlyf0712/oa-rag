# 03 — Append-only expansion with query-consistency gate

**What to build:** Graph expansion feeds `Searcher.search(graph_expand=1)` as a confidence boost. From the top-3 base results, it walks the four traversal-eligible structural edge types (`same_conversation`, `sender_receiver`, `same_sender`, `same_company`), one hop, and appends relevant neighbor documents below the base ranking. Each expanded document is scored as `parent_score × 0.8 × neighbor_query_relevance`, where the neighbor's own hybrid relevance to the segmented query is computed with the already-loaded index — so a structurally-connected but query-irrelevant neighbor is balanced out instead of blindly followed.

**Blocked by:** 02 — Purely structural graph invariant

**Status:** ready-for-agent

- [ ] Expansion is append-only: base results are never reordered or removed; expanded documents appear below them and the list is truncated to the requested limit
- [ ] Expansion walks only the four traversal-eligible edge types; `same_label` is never traversed
- [ ] Expanded documents are re-checked against the active label/date filters before appearing
- [ ] Every expanded document is annotated with `_graph_relation` and `_from_node` provenance
- [ ] The query-consistency gate is applied: an expanded neighbor's score includes its own hybrid relevance to the query; a neighbor irrelevant to the query does not surface in the top results
- [ ] All knobs (`max_expand` seeds, `hop_discount`, traversal-eligible edge set) are parameters on `_graph_expand`
- [ ] Tested through `Searcher.search()`: expansion keeps base relevance, surfaces genuine same-conversation connections, and never tampers with the base order