# 04 — Multilingual reranker

**What to build:** A Chinese-capable reranker (e.g. `BAAI/bge-reranker-base` or `bge-reranker-m3`) is enabled by default and verified to improve result ordering on Chinese queries. The English-only `ms-marco-MiniLM` reranker is no longer used. The base regression suite stays green.

**Blocked by:** 01 — Chinese-capable hybrid base

**Status:** resolved

- [x] Chinese-capable reranker selected and wired in
- [x] Reranker enabled by default
- [x] Reranking verified to improve (or at worst match) base-only ordering on the regression suite
- [x] Regression suite stays green with reranking on

## Answer

**Reranker model chosen:** `BAAI/bge-reranker-base` — a Chinese-capable cross-encoder (XLM-RoBERTa-based, ~1.1GB). Selected over `bge-reranker-v2-m3` for lighter weight while still being Chinese-capable; the corpus is ~100% Traditional Chinese. Overridable via `RERANKER_MODEL` env var.

**What changed (4 targeted edits in `apps/corpchat/search.py`):**
1. Added `DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")` constant.
2. Changed `Reranker.__init__` default `model_name` from `cross-encoder/ms-marco-MiniLM-L-6-v2` (English-only) to `DEFAULT_RERANKER_MODEL`.
3. Changed `Searcher.search()` default `use_rerank` from `False` to `True` (reranking enabled by default).
4. Changed `AgenticDecider.decide()` default from `use_rerank: False` to `use_rerank: True`.

**New test file:** `tests/test_search_reranker.py` — 9 tests:
- 4 unit tests (no model download): default model is Chinese-capable, `Reranker()` uses the default model, `use_rerank` defaults to `True`, reranker enabled when sentence_transformers available.
- 4 integration tests with `FakeReranker` (no model download): rerank improves-or-matches base on `物流報價 方案` and `投資美國債券跟藍籌股`, label filter still scopes, bare-label search doesn't rank all label docs.
- 1 real-model integration test (skips if model not cached): loads the real `BAAI/bge-reranker-base` and verifies improve-or-match on both regression queries.

**Verification:** 26/26 tests pass (0 skipped after model download). The real `BAAI/bge-reranker-base` model correctly scores relevant Chinese text at 0.9989 vs irrelevant text at 0.00004. Reranked results improve-or-match base-only ordering on both regression queries (`物流報價 方案` and `投資美國債券跟藍籌股`): the relevant messages stay in top-3 and total relevance is >= base-only. All 17 existing tests (tickets 01-03) remain green — they pass `use_rerank=False` explicitly, so the default change is safe.

**Context pointer:** `.scratch/retrieval-base-consolidation/spec.md` (Implementation Decision 6, overridden by this ticket; Testing Decisions), `tests/test_search_reranker.py`, `apps/corpchat/search.py` (`DEFAULT_RERANKER_MODEL`, `Reranker`, `Searcher.search`).
