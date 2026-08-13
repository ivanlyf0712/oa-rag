# 03 — Ask UI: merged results table with rank-by-risk

**Spec:** `docs/specs/unified_search_risk_merge.md`

**What to build:** one results table for all Ask queries, rendered from the result store (ticket 02). Controls beside the table: (a) a **"Rank by risk" toggle** that switches sort to risk score descending and reveals risk columns plus the min-score control — auto-enabled when the planner detects risk intent, user-overridable; (b) a **minimum risk score** number input, default 80, visible only in rank-by-risk mode, filtering displayed rows client-side (general mode never hides rows); (c) a **column picker** multiselect over all flattened/risk columns (Browse-style dynamic columns) with a small sensible default set per mode. Result count always shows total matched vs displayed. The separate risk-results renderer (`_render_risk_evidence`) is removed; per-contract detail selectbox stays and feeds the rebuilt detail view (ticket 04).

**Blocked by:** 02 — LLM observation formatter, result store & agent wiring.

**Status:** done

- [x] Single table renders all matched rows from the result store (no cap on displayed rows)
- [x] Rank-by-risk toggle re-sorts by risk score desc; risk-intent queries open with it ON
- [x] Min-score input (default 80) filters only in risk mode; general mode shows everything
- [x] Column picker adds any flattened/risk column; defaults differ sensibly per mode
- [x] Result count displays total matched vs displayed
- [x] `_render_risk_evidence` and its regex parsing gone; detail selectbox wired to ticket 04 view
- [x] Pure helpers (sort/filter/column resolution) unit-tested; full flow manually verified in Docker
