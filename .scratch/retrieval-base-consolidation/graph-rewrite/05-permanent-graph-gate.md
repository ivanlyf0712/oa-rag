# 05 — Permanent graph gate and improve-or-match bar

**What to build:** A permanent test module asserts the graph layer end-to-end through `Searcher.search()` with `graph_expand=1`: the structural-only invariant, base relevance preserved under expansion, genuine same-conversation connections surfaced, label protection under expansion, and the non-tampering / improve-or-match bar. It also includes the "useless connection balanced out" test — a structurally-connected neighbor that is irrelevant to the query must not surface. The existing ticket-01 and ticket-02 suites remain untouched and stay green.

**Blocked by:** 03 — Append-only expansion with query-consistency gate, 04 — Graph lifecycle cleanup

**Status:** ready-for-agent

- [ ] A permanent test module exercises the graph through `Searcher.search()` with `graph_expand=1` on the deterministic conversation-template fixture
- [ ] Structural-only invariant asserted (every node ≥1 structural edge; no vector-similarity edges)
- [ ] Expansion keeps base relevance: `graph_expand=1` top-3 still contains the logistics (`物流`+`報價`) and investment (`債券`+`藍籌`) messages, matching `graph_expand=0`
- [ ] Genuine connection surfaces: a same-conversation message missed by base-only top-3 appears under `graph_expand=1`, annotated with `_graph_relation`
- [ ] Label protection under expansion: bare-label search with `graph_expand=1` does not rank all docs of that label; label-filtered search with `graph_expand=1` contains only that label
- [ ] Non-tampering / improve-or-match bar: the top-3 base documents are present in the same relative order with `graph_expand=1`, and graph expansion adds a relevant same-conversation document for at least one query
- [ ] Query-consistency gate: a structurally-connected neighbor irrelevant to the query is balanced out and does not surface in the top results
- [ ] `pytest tests/ -v` is fully green, including the unmodified ticket-01 and ticket-02 regression suites