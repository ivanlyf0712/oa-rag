# Unified Search & Risk Merge

**Spec status:** ready-for-agent
**Source:** grilling session 2026-08-13; all decisions below were explicitly confirmed by the user.

## Problem Statement

The Ask view runs two divergent pipelines behind one input box:

- **Contract search** feeds the LLM only `ref / counterparty / type + 200 chars of chunk text`. Dates, amounts and status never reach the model — "is CCA20250096 expired" gets a useless answer even though the dates sit in the index metadata.
- **Risk search** is a separate tool with its own formatter, its own gate (`gate_and_sort` silently drops contracts below a minimum score), its own LLM planner gate (if the planner says "not a risk query", risk data never exists), and a UI that **regex-parses the LLM observation string** to rebuild its table.

Two formatters, two tables, two silent-degradation points. Retrieval is also silently capped at 10 rows, so larger result sets are truncated without the user knowing. Meanwhile the contract detail tabs confuse: the Decoded tab only shows coded fields *that have values*, so sparse contracts render partial or empty JSON that looks broken.

## Solution

One search pipeline, one structured row type, one table, one detail view:

1. **Unified search service** — a single function that: extracts structured filters (LLM planner, shared by fast-routes and ReAct), builds the candidate set (all contracts passing structured filters, uncapped; or generous ~50-contract semantic retrieval), **always computes risk score/severity/signals** (deterministic, cheap), and ranks by relevance or risk. It returns structured `ContractRow` dicts — text is formatted only at the LLM boundary.
2. **Bounded, richer LLM observation** — per-row fields now include dates, amount, status, and risk data; capped at ~50 rows with an explicit "+N more contracts not shown" overflow marker. The UI table and detail views always render the full result set; only the LLM prompt is budgeted (the shared proxy degrades under large prompts).
3. **One results table** with a **"Rank by risk" toggle**: general view sorts by relevance and shows all matches; risk view sorts by risk score and applies a **visible, user-adjustable minimum-score filter (default 80)**. Risk-intent queries ("show risky contracts") auto-toggle rank-by-risk on. A column-picker (Browse-style) lets the user add any flattened/risk column on demand.
4. **Detail tabs rebuilt as three lenses over one record**: **Raw** = the complete stored record (all ~155 columns) as JSON; **Contextual** = only self-explanatory fields (plain values + coded fields rendered decoded, e.g. "unlimited liability: Yes/No"; opaque IDs and undecodable codes excluded); **Risk** = risk view derived from the contextual fields. Missing values render as explicit placeholders instead of silently absent keys.
5. **Architecture**: LangGraph ReAct stays (multi-tool future). Tools become thin adapters over the unified service; a per-query **result store** carries the structured rows to the UI (adopted from corpchat's `snapshot_meta` pattern), deleting the regex observation-parser.

## User Stories

1. As a contracts reviewer, I want the LLM's aggregate answer to see contract dates, amounts, and status, so that questions like "is CCA20250096 expired" get substantive answers.
2. As a contracts reviewer, I want to see every contract matching my structured query, so that no matches are silently truncated.
3. As a contracts reviewer, I want semantic queries to return a generous ranked set (~50 contracts) rather than a silent top-10, so that broad questions surface the full relevant tail.
4. As a contracts reviewer, I want an explicit overflow note when the LLM only saw part of the result set, so that I know to consult the table for the remainder.
5. As a risk officer, I want every search result to carry a risk score, so that I can re-rank any result set by risk without issuing a new query.
6. As a risk officer, I want a "rank by risk" toggle beside the results table, so that I can switch between relevance and risk ordering in one click.
7. As a risk officer, I want risk-intent queries to open in risk-ranked mode automatically, so that "show risky contracts" does what I mean without extra clicks.
8. As a risk officer, I want an adjustable minimum-risk-score filter with a sensible default (80), so that I can widen or narrow the risk view myself — and nothing is ever hidden by an invisible server-side gate.
9. As a contracts reviewer, I want non-risk queries to always show all matches, so that ranking choice never hides results from me.
10. As a risk officer, I want to add risk columns (score, severity, signals) and any other flattened/risk column to the results table via a column picker, so that I can inspect exactly the fields I care about without a 28-column wall of text by default.
11. As a contracts reviewer, I want the detail view's Raw tab to show every column of the stored record as JSON, so that no field is ever invisible to me.
12. As a contracts reviewer, I want the Contextual tab to show only human-meaningful fields — plain values plus decoded yes/no labels — so that I'm not staring at opaque IDs and codes.
13. As a contracts reviewer, I want empty fields to render as explicit "(empty)" placeholders, so that a sparse contract doesn't look like a broken app.
14. As a risk officer, I want the Risk tab derived from the same normalized fields I see in Contextual, so that the three tabs never contradict each other.
15. As a developer, I want tools to return structured rows and the UI to render from a result store, so that no UI code ever regex-parses an LLM observation string again.
16. As a developer, I want one planner shared by fast-routes and the ReAct agent, so that filter extraction can't diverge between paths.
17. As a developer, I want risk scoring to be unconditional and deterministic, so that the LLM planner can never gate risk data out of existence.
18. As an operator, I want the LLM prompt bounded (~50 rows), so that the shared proxy stays healthy and the "LLM unreachable" banner doesn't return on large result sets.
19. As a developer (phase 2), I want a `contracts_where` SQL tool ported from corpchat's `search_messages_where`, so that exact structured questions ("list all contracts over HK$5M ending before 2027") are grounded in the database rather than semantic guesswork.

## Implementation Decisions

### Unified search service (module: `apps/search/service.py` + `apps/risk_search.py`)

- Single entry point, signature: `search(query, filters=None, *, rank_by="relevance", limit=None) -> List[ContractRow]`.
- `ContractRow` carries: identity (ref_no, title, counterparty, contract_type), dates (start/end), amount (+ label), status (+ label), risk (score, severity, matched_signals, explanation), best-matching snippet text, and the full per-contract record reference for detail rendering.
- **Planner moves inside the service**: filter extraction (contract type, date ranges, risk intent, rank hint) happens in one place, used by both the regex fast-routes and the ReAct tool adapter. The RiskPlanner "is this a risk query?" mode gate is removed; its filter-extraction capability is preserved and generalized to all queries.
- **Candidate set semantics**:
  - Exact ref_no lookups: unchanged (deterministic bypass).
  - Structured/exact-filter queries: **all** matching contracts, no cap.
  - Semantic queries: generous retrieval (~50 contracts after chunk-to-contract dedupe), ranked by relevance.
- **Risk scoring is unconditional**: `score_risk` runs over every candidate set. `gate_and_sort` is retired as a result-shaping step; min-score filtering becomes a UI-layer concern only.
- Ranking: `rank_by="relevance"` preserves semantic/exact order; `rank_by="risk"` sorts by risk score descending.

### LLM observation (module: `apps/search/service.py` formatter + `apps/search/langchain_agent.py`)

- Per-row observation fields: ref, counterparty/title, type, status label, start/end dates, amount, risk score/severity, top signals, ~200-char snippet.
- **Budget**: first ~50 numbered rows enter the prompt; if more matched, an explicit `"(+N more contracts not shown - see results table)"` marker is appended.
- The existing 5-line truncation in `_default_synthesize` is superseded by this formatter (budget enforced at the formatter, not by line-count hack).
- Synthesis prompt instructions unchanged (concise summary, no ID dumps).

### Agent & tools (module: `apps/search/langchain_agent.py`, `apps/search_cli.py`)

- LangGraph ReAct **stays**. `search_contracts` becomes a thin adapter over the unified service; the separate risk tool is retired (one tool, one pipeline).
- **Result store** (new small module, pattern adopted from corpchat's `tools.snapshot_meta`): tool adapters stash the structured `ContractRow` list + query meta per invocation; the Streamlit app renders tables from the store. The regex observation-parser in the app is deleted.
- `fallback` semantics unchanged; the loud LangGraph-unavailable warning stays.

### Ask UI (module: `apps/app.py`)

- One results table for all queries. Controls beside it:
  - **"Rank by risk" toggle** — switches sort to risk score desc and reveals risk columns + min-score control. Auto-enabled when the planner detects risk intent; user can switch it back.
  - **Minimum risk score** number input, default **80**, visible only in rank-by-risk mode; filters displayed rows client-side (nothing hidden in general mode).
  - **Column picker** multiselect over all flattened/risk columns (Browse-style dynamic columns); sensible small default set per mode.
- Result count always displayed, including total matched vs shown.
- Per-contract detail selectbox unchanged in placement; feeds the rebuilt detail view.

### Detail tabs (module: `apps/app.py` detail renderer)

- **Raw** (replaces Decoded): the complete stored `raw` record (all ~155 columns) rendered as JSON. No curation, no omissions.
- **Contextual**: curated self-explanatory view = plain-value fields (identity, dates, amounts, names, workflow state) + the 20 coded fields rendered as decoded labels ("Yes/No"), grouped per `data/contract-data-column-groups.md`; audit/system identifiers and undecodable codes excluded (they live in Raw). Missing values show an explicit "(empty)" placeholder.
- **Risk**: score, severity, and matched signals with human-readable labels, derived from the same normalized record backing Contextual (single source of truth — tabs cannot disagree).

### Phase 2 (separate ticket, out of this spec's critical path)

- `contracts_where(condition)` tool ported from corpchat's `search_messages_where`: natural-language condition → validated SQL over the sections SQLite DB (LLM translation with rule-based fallback), returning the same `ContractRow` set into the same result store. Primary use case: "list all contracts with <condition>". A bare "list all" (empty/vacuous condition) means no WHERE clause and returns every contract — subject to the standard 50-row LLM budget with overflow marker, while the table shows everything. "list all"-style queries with no semantic content always route to the structured path (this tool or the service directly), never to vector search.

## Testing Decisions

- **Primary seam: the unified search service** — pure Python over the sections DataFrame + Searcher; no Streamlit, no LLM. All existing service and risk-search tests retarget here. New tests: uncapped structured retrieval, ~50-contract semantic bound, unconditional risk scoring on every candidate set, relevance-vs-risk ordering.
- **Secondary seam: the observation formatter** — pure function rows → text. Tests: richer fields present (dates/amount/status), 50-row budget respected, overflow marker exact, empty-set behavior.
- **Tertiary seam: result store** — stash/snapshot round-trip, per-invocation isolation.
- **Planner**: filter extraction from representative queries (type, date range, risk intent detection) with the LLM mocked, following the existing RiskPlanner test pattern.
- UI (toggle, min-score control, column picker, tab rendering) is verified manually per project convention; logic that can be pure (placeholder filling, contextual field curation) is factored into testable helpers.
- Prior art: `tests/test_search_service.py`, `tests/test_risk_search.py`, `tests/test_langchain_agent.py`, corpchat's `tests/test_tools_expansion.py` for the result-store pattern.

## Out of Scope

- **Truly unbounded LLM prompts / map-reduce summarization** — explicitly rejected in grilling; bounded-with-overflow chosen instead.
- **Changes to risk weights or severity tiers** — scoring config is untouched; only its *application* becomes unconditional.
- **OCR / attachment content extraction** — unchanged.
- **Browse tab redesign** — the column-picker pattern is borrowed from it, not rebuilt.
- **Index rebuild** — not required: the full `raw` record is already stored in the index metadata (`core/db.py`), so Raw/Contextual tabs are UI-side work.
- **`contracts_where` SQL tool** — specified here for architecture context only; implemented as a phase-2 ticket after the merge lands.
- **CCA20260252 attachment re-sync** — separate pending operational task.

## Further Notes

- **Corpus-scale assumption**: uncapped retrieval and always-on risk scoring are priced for the current ~260-contract corpus. If the corpus grows by an order of magnitude, revisit caps (make them config-driven rather than deleting them).
- **corpchat-rag provenance**: the result-store (`snapshot_meta`), the SQL-condition tool (`search_messages_where` with `_validate_sql` + LLM fallback), and the per-process LLM-availability cache are all adopted from corpchat-rag, where they are already tested and in production use.
- **The "sometimes fields are not showing" bug** turned out to be by-design omission (`_build_decoded_fields` / `_build_contextual_fields` skip NULL/empty fields at index time). The three-lens rebuild fixes it at the rendering layer by sourcing from the complete `raw` record with explicit placeholders.
- **Default min-score 80** was calibrated against the live corpus: score distribution max 145, ≥80 → 8 contracts, ≥50 → 16, ≥20 → 104. 80 means "the heavy hitters"; the control is user-adjustable precisely because the right cutoff is a judgment call.
