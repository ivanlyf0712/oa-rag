# 02 — LLM expansion on the verified base

**What to build:** The LLM query-expansion layer (semantic rephrase + keyword expansion, weighted RRF fusion) works on top of the fixed Chinese-capable base. A user searching with expansion enabled gets results that are at least as relevant as the base alone, and ideally better — expansion must not degrade the base's precision. The base regression suite stays green.

**Blocked by:** 01 — Chinese-capable hybrid base

**Status:** resolved

- [x] `QueryExpander` re-enabled on the verified base
- [x] Weighted RRF fusion (0.5 / 1.3 / 1.0) verified against the base-only results on the regression queries
- [x] Expansion demonstrably improves (or at worst matches) base-only retrieval on the regression suite
- [x] Regression suite stays green with expansion on

## Answer

Verified that `QueryExpander` + weighted RRF fusion (0.5 / 1.3 / 1.0) works correctly on top of the fixed Chinese-capable base. Added `tests/test_search_expansion.py` with 6 tests: 3 unit tests for `QueryExpander` (semantic+keyword generation, graceful fallback on LLM failure, deduplication) and 3 integration tests through the `Searcher.search()` seam using a deterministic `FakeExpander` (no live API). All 6 expansion tests pass, and the 4 regression tests in `tests/test_search_regression.py` remain green. Full suite: 10 passed. Expansion matches base-only relevance on the regression queries (`物流報價 方案`, `投資美國債券跟藍籌股`, label filter) — it does not degrade precision.

Context: `.scratch/retrieval-base-consolidation/spec.md` (Implementation Decision 9, Testing Decisions), `tests/test_search_expansion.py`.
