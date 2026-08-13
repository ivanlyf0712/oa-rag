# 01 — Unified search service

**Spec:** `docs/specs/unified_search_risk_merge.md`

**What to build:** one search pipeline replacing the contract/risk split. `ContractSearchService.search(query, filters=None, *, rank_by="relevance", limit=None) -> List[ContractRow]` where each row carries identity (ref_no, title, counterparty, contract_type), dates (start/end), amount (+ label), status (+ label), risk (score, severity, matched_signals, explanation), best-matching snippet, and the per-contract record reference. Filter extraction (contract type, date range, risk intent, rank hint) moves inside the service via a shared planner used by both fast-routes and the ReAct adapter; the RiskPlanner mode gate is removed (its filter-extraction is generalized to all queries). Candidate semantics: exact ref lookups unchanged; structured/exact-filter queries return ALL matches uncapped; semantic queries return ~50 contracts after chunk-to-contract dedupe, ranked by relevance. `score_risk` runs unconditionally over every candidate set; `gate_and_sort` is retired as a result-shaping step. `rank_by="risk"` sorts by score descending; `"relevance"` preserves retrieval order.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] Single `search()` entry point returns structured `ContractRow` dicts with all fields listed above
- [x] Planner (filters + risk intent + rank hint) lives in the service, shared by all entry paths; RiskPlanner mode gate gone
- [x] Structured/exact-filter queries return all matches (no cap)
- [x] Semantic queries return ~50 contracts post-dedupe, relevance-ranked
- [x] Risk score/severity/signals computed for every candidate in every search
- [x] `gate_and_sort` no longer drops rows from any result set
- [x] `rank_by="risk"` sorts desc; `"relevance"` preserves retrieval order
- [x] Existing service + risk tests retargeted to the unified seam; new tests for uncapped structured retrieval, semantic bound, unconditional scoring, both orderings
