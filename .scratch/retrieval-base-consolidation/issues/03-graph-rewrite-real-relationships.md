# 03 — Graph rewrite on real conversation relationships

**What to build:** The graph layer is rebuilt on genuine conversation structure — sender, receiver, chatroom (`open_kfid`), company — instead of vector similarity. A user searching with graph expansion enabled gets results that surface real connections (e.g. who talks to whom about what), not a similarity echo. The base regression suite stays green.

**Blocked by:** 01 — Chinese-capable hybrid base

**Status:** ready-for-agent

- [ ] Graph nodes/edges encode real relationships: sender, receiver, chatroom, company, label
- [ ] Graph expansion surfaces genuine connections rather than vector-similarity neighbors
- [ ] Graph expansion verified to improve (or at worst match) base-only retrieval on the regression suite
- [ ] Regression suite stays green with graph expansion on

## Notes

This is the design-heavy ticket. It may need its own grilling/spec session before implementation — the current graph is built on vector similarity, and the redesign to real conversation edges is a design task, not a bug fix.