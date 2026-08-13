# Spec: Unified Tag-Filter Search + Grounded Answer Generation

> Status: ready-for-agent
> Supersedes: the `risk_search` tool/mode fork (partially), and the exact-ref band-aids in `app.py` / `search_cli.py`.
> Index under test: `apps/search_index` (268 sections, 259 distinct `ref_no`; `CCA20250096` = `contract_109__chunk0`).

## Problem Statement

A contract-screening user types natural-language questions into the Streamlit agentic UI. Three things go wrong today:

1. **Exact lookups are unreliable.** Typing a reference number like `CCA20250096` is routed to the generic semantic "Contract search (normal)" path, where the correct contract surfaces with a hybrid score of `0.0000` and can land at rank 5 — or the pipeline crashes outright (`'Searcher' object has no attribute '_search_exact_ref'`). The Streamlit evidence panel then renders a bare `contract_109__chunk0` with `score None` and empty decoded/contextual metadata.
2. **Tag-based filtering doesn't work.** "Find completed contracts with high risk" should filter on the `status` tag and the risk flags, but the semantic path ignores structured tags, and the separate "risk search" mode is a fragile parallel stack.
3. **Answers can't be trusted.** The generated summary free-styles over whatever it is given, so it can contradict the authoritative tag values without flagging it.

The user's bottom line: accurate data about a *particular* contract must be recognized and used correctly, every time.

## Solution

One unified pipeline for all queries:

- A lightweight LLM **recognizes every tag mentioned in the query** — metadata tags (`ref_no`, `title`, `status`, `contract_type`, dates, amounts) and risk tags (`IsRisksAccepted`, `Over5M`, `FlagNeedLegal`, `unlimitedliabilitiesorliabilit`, …) are treated identically — and emits a **flat list of `{field, op, value}` filter clauses**.
- Those clauses are applied as a **deterministic SQL/pandas filter over `sections.tags`**, bypassing txtai ranking entirely. Exact `ref_no`/`title` matches are validated against real database values before retrieval, so the returned contract is guaranteed to be the one the user named.
- **Every result row carries an inferred `risk_level` column**, computed by the existing `score_risk` weights/tiers. There is no separate "risk mode" — risk is data, not a route. "High risk" in a query filters on the *same* scoring function and threshold that produces the displayed `risk_level`, so filter and display can never disagree.
- **Answer generation is grounded:** the verdict (risk level, status, field values) comes from tags and is stated plainly; the LLM only reports supporting facts from the chosen contract attachment and **flags contradictions** between attachment text and tags without judging or overriding them.
- **Semantic hybrid search (txtai) is reserved** for genuine free-text conceptual questions (e.g. "contracts mentioning unlimited liability clauses") that no tag captures.

## User Stories

1. As a contract reviewer, I want to type a reference number like `CCA20250096` and get exactly that contract with its full metadata, so that I can trust I'm looking at the right record.
2. As a contract reviewer, I want a ref embedded in a sentence ("is contract CCA20250096 high risk?") to resolve to that exact contract, so that I don't have to reformat my question.
3. As a contract reviewer, I want to search by exact title (e.g. `CKTEST080604`), so that I can find contracts whose reference number I don't remember.
4. As a compliance officer, I want every search result to show a consistent `risk_level` column, so that I can scan risk at a glance regardless of how I phrased the query.
5. As a compliance officer, I want "high risk" in my query to filter using the same scoring rule as the displayed risk level, so that the filter and the labels never contradict each other.
6. As a contract reviewer, I want to combine tags freely ("completed contracts over 5M related to data"), so that I can slice the portfolio without learning a query language.
7. As a compliance officer, I want the generated answer's verdict to come from the deterministic tags, so that the answer is reproducible and auditable.
8. As a compliance officer, I want the answer to point out when the contract document contradicts the tags, so that data-quality issues surface instead of being silently smoothed over.
9. As a user, I want semantic questions about contract content to still work, so that I can find clauses that aren't captured by any tag.
10. As a contract reviewer, I want the supporting evidence panel to show decoded and contextual fields fully populated, so that I can verify why a result matched.
11. As a user, I want the system to never crash on a title or ref-in-sentence query, so that I can rely on the tool in front of colleagues.
12. As an operator, I want one query pipeline instead of parallel contract/risk stacks, so that routing bugs can't strand my query in the wrong tool.



## Implementation Decisions

### Architecture: one pipeline, three stages

- **Stage 1 — Tag recognition (LLM, lightweight).** A single planner prompt maps the user query to `{filters: [{field, op, value}], semantic_query: string|null}`. All recognized tags — metadata and risk alike — go into the same flat `filters` list; there is no "risk mode" flag. If nothing tag-like is recognized, `semantic_query` carries the raw question to Stage 2b. Extracted `ref_no`/`title` values are validated against actual column values in the index; unrecognized identifiers fall back to semantic search rather than fabricating a match.
- **Stage 2a — Deterministic tag filter.** When `filters` is non-empty, results come from filtering `sections.tags` (SQL/pandas) on those clauses — no txtai, no ranking, no rerank, no chunk expansion. Exact-identifier queries (`ref_no`/`title` equality) return precisely the matching contract row(s).
- **Stage 2b — Semantic hybrid search.** Only when `filters` is empty does the query go through the existing txtai hybrid path, returning clause-level passages.
- **Stage 3 — Enrichment + grounded generation.** Whatever the route, result rows are passed through the existing `score_risk` function to attach `risk_score`/`risk_severity` (displayed as `risk_level`) and `matched_signals`. The answer generator is given the tag verdict verbatim and instructed to state it plainly, report attachment facts, and flag tag-vs-document contradictions without judging.

### Modules modified

- **Query planner** (currently `RiskPlanner` in the risk-search module): generalized from risk-only to all-tags recognition; the `mode` (risky/general/clarify) concept is removed in favour of "filters or semantic". The allowlist of filterable fields grows to cover metadata tags (`ref_no`, `title`, `status`, `contract_type`, `department`, dates, amounts) alongside the existing risk fields; values are normalized (yes/no/na for flags; decoded labels for enums like status).
- **Searcher** (search core): gains a deterministic tag-filter path over the index's `sections` table (equality + decoded-label matching), used by Stage 2a. The broken exact-ref shortcut (AttributeError on `_search_exact_ref`) is removed/replaced by this path. The semantic path is hardened so a missing graph extra (GrandCypher ImportError) can no longer crash title/ref-in-sentence queries.
- **Agent layer** (both the manual ReAct agent and the LangChain agent): the `risk_search` tool is deleted; a single contract-search tool accepts the planner's filter list. Both agents share **one route-decision function** (the planner), so tool routing cannot diverge between agents.
- **Streamlit app**: the evidence re-run uses the same unified filter path (no app-level exact-ref special-casing); evidence rows are normalized to the `{id, text, score, metadata}` shape so decoded/contextual fields render fully; the results table always includes the `risk_level` column.
- **Answer generation**: the synthesis prompt contract changes to include (i) the deterministic verdict computed from tags, (ii) attachment summary facts, and (iii) an explicit instruction to flag contradictions without adjudicating them.
- **Attachment analysis** (per-contract attachment selection/extraction/summarization) stays as-is and plugs into Stage 3 unchanged.

### Data notes (from live inspection)

- `sections.tags` carries flat keys (`ref_no`, `title`, `status`, `contract_type`, …) plus `decoded_fields` as `{field: {raw, label}}` and `contextual_fields`; enum tags like `status`/`contract_type` are raw ints whose labels live in decoded form — filters must match on decoded labels.
- `score_risk` already computes `risk_score`, `risk_severity`, `matched_signals`, `risk_explanation` from configurable weights/tiers; `gate_and_sort` provides the high-risk threshold used for both filtering and labelling.

## Testing Decisions

- **Good tests assert external behaviour only**: query in → route/filters produced → rows out (with `risk_level`) → answer contract honoured. No assertions on internal ranking internals, prompt wording beyond the verdict/flag contract, or LLM provider details.
- **Highest seams, reused from existing tests**:
  - *Planner seam* (existing `test_contract_router.py`, `test_contract_agent.py`): stub the LLM; assert a query maps to the right filter clauses (ref query, title query, status+risk query, pure-semantic query). One shared planner used by both agents is asserted directly.
  - *Search seam* (existing `Searcher.search` tests in `test_oa_app.py`): against a stubbed `sections` table, assert (a) exact `ref_no` returns exactly the matching row with full metadata; structured multi-tag filters AND correctly; semantic fallback engages only when no filters exist; no crash on title/ref-in-sentence.
  - *Risk scoring seam* (existing `test_risk_search.py`): assert the same scoring/threshold drives both the "high risk" filter and the `risk_level` label.
  - *Generation seam*: stub the LLM; assert the prompt carries the tag-derived verdict and the contradiction-flagging instruction, and the output states the verdict plainly.
  - *UI seam* (existing `test_oa_app.py` render tests with fake `st`): assert evidence renders decoded/contextual fields fully and the table includes `risk_level`.
- **Acceptance tests (the bar agreed in design)** — all four must pass reproducibly against `apps/search_index`:
  1. `CCA20250096` → exactly `contract_109`, full metadata.
  2. "is contract CCA20250096 high risk?" → resolves to contract 109; verdict from its tags; attachment facts cited.
  3. "find completed contracts with high risk" → filters `status=completed` + high-risk threshold, sorted by score, no semantic drift.
  4. "contracts mentioning unlimited liability" → hybrid semantic search returns relevant clause passages.
- **Prior art**: `tests/test_oa_app.py`, `test_risk_search.py`, `test_contract_router.py`, `test_contract_agent.py`, `test_attachment_summary.py` already establish the monkeypatch + stub-LLM + stub-`sections`-table style to follow.

## Out of Scope

- Rebuilding or re-chunking the index; changing the embedding model.
- Changing the risk weights/tiers themselves (configuration, not code).
- The attachment selection priority/extraction logic (already built and tested).
- Multi-contract comparison queries ("compare risk across departments") beyond simple tag filters.
- Authentication, permissions, audit logging.
- Removing the clarify fallback for genuinely ambiguous queries (kept).

## Further Notes

- The separate `risk_search` tool, its mode-based planner contract (`risky_contracts`/`general_search`), and the app-level exact-ref helper are all superseded and should be deleted once the unified path lands, to prevent regression via parallel stacks.
- The semantic path's GrandCypher dependency crash should be fixed defensively regardless of route, since semantic remains the fallback for unrecognized identifiers.

