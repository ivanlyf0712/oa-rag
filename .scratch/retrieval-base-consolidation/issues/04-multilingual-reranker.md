# 04 — Multilingual reranker

**What to build:** A Chinese-capable reranker (e.g. `BAAI/bge-reranker-base` or `bge-reranker-m3`) is enabled by default and verified to improve result ordering on Chinese queries. The English-only `ms-marco-MiniLM` reranker is no longer used. The base regression suite stays green.

**Blocked by:** 01 — Chinese-capable hybrid base

**Status:** ready-for-agent

- [ ] Chinese-capable reranker selected and wired in
- [ ] Reranker enabled by default
- [ ] Reranking verified to improve (or at worst match) base-only ordering on the regression suite
- [ ] Regression suite stays green with reranking on