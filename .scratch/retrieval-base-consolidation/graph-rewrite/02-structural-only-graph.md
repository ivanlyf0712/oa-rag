# 02 — Purely structural graph invariant

**What to build:** The graph is verified to be purely structural — every chunk node has at least one structural edge, so txtai's vector auto-inference never runs, and no vector-similarity edges exist in the graph.

**Blocked by:** 01 — Structural payload into the graph

**Status:** ready-for-agent

- [ ] Every graph node has at least one edge (the `same_label` edges guarantee this even for single-chunk conversations)
- [ ] Because txtai graph inference defaults to `approximate: true`, nodes that already have edges are skipped by vector inference — verified by test
- [ ] No vector-similarity edges exist in the graph (the purely structural invariant is asserted by test)
- [ ] The graph is fully deterministic: rebuilding the same fixture produces the same graph