# 04 — Graph lifecycle cleanup

**What to build:** The graph lifecycle is simplified to match the structural design: `graph_mode=auto` and `graph_mode=llm` both resolve to the same structural construction, `graph_mode=off` disables the graph, and the dead/non-deterministic LLM relation-extraction path is removed. The `same_label` edge remains recorded but never traversed.

**Blocked by:** 02 — Purely structural graph invariant

**Status:** ready-for-agent

- [ ] `graph_mode=auto` and `graph_mode=llm` both produce the same structural graph; `graph_mode=off` produces no graph
- [ ] The dead LLM relation-extraction code path (which called a non-existent txtai API and was non-deterministic) is removed
- [ ] The misleading "vector auto-inference" log message is removed/replaced to reflect the structural graph
- [ ] `same_label` edges are recorded in the graph but confirmed never traversed by expansion
- [ ] No base retrieval behaviour changes as a result of this cleanup