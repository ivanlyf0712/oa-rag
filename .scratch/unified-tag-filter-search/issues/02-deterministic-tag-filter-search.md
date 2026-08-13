Status: ready-for-agent
Type: task

# 02 — Deterministic tag-filter search in Searcher + `risk_level` enrichment

**What to build:** the `Searcher` gains a deterministic tag-filter path over the index's `sections` table. When the planner (ticket 01) returns a non-empty filter list, results come from filtering `sections.tags` on those clauses — equality plus decoded-label matching, AND-combined — with no txtai, no ranking, no rerank, no chunk expansion. Exact-identifier queries (`ref_no`/`title` equality) return precisely the matching contract row(s) with full metadata. Result rows are then passed through the existing `score_risk` function so every row carries `risk_score`, `risk_severity` (surfaced as the `risk_level` column) and `matched_signals`. This also removes/replaces the broken exact-ref shortcut that raised `'Searcher' object has no attribute '_search_exact_ref'`. When the filter list is empty, the existing hybrid semantic path runs unchanged.

**Blocked by:** 01 — Unified tag planner.

## Acceptance criteria
- An exact `ref_no` filter returns exactly the matching contract row (e.g. `CCA20250096` → `contract_109`) with full metadata — never a semantic-ranking miss.
- Structured multi-tag filters AND together correctly across metadata and risk fields.
- Every returned row carries `risk_score`, `risk_severity`/`risk_level`, and `matched_signals` computed by `score_risk`.
- Enum tags such as `status`/`contract_type` match on decoded labels, not raw ints.
- The exact-ref AttributeError no longer occurs for any ref-like or title-like query.
- When `filters` is empty, the hybrid semantic path is used and behaves as before.
- All verified against a stubbed `sections` table — no live index or database required.
