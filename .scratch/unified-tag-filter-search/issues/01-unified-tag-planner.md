Status: ready-for-agent
Type: task

# 01 — Unified tag planner: recognize all tags → flat filter list

**What to build:** a single, pure query-planner that maps any natural-language query to a flat list of filter clauses plus an optional semantic fallback. It treats metadata tags (`ref_no`, `title`, `status`, `contract_type`, dates, amounts) and risk flags (`IsRisksAccepted`, `Over5M`, `FlagNeedLegal`, `unlimitedliabilitiesorliabilit`, …) identically — there is no "risk vs general" mode fork. Output shape: `{filters: [{field, op, value}], semantic_query: string|null}`. Extracted `ref_no`/`title` values are validated against real values in the index; unrecognized identifiers fall back to `semantic_query` rather than fabricating a match. When nothing tag-like is recognized, `filters` is empty and `semantic_query` carries the raw question. A stub LLM makes the whole function unit-testable.

**Blocked by:** None — can start immediately.

## Acceptance criteria
- A ref-number query (e.g. `CCA20250096`) yields a single `ref_no` equality filter, validated against the index; an unknown ref yields a semantic fallback, not a fabricated filter.
- A title query (e.g. `CKTEST080604`) yields a `title` equality filter.
- A combined query ("completed contracts over 5M related to data") yields multiple AND-combined filter clauses across metadata and risk tags in one flat list.
- A pure conceptual query ("contracts mentioning unlimited liability") yields empty `filters` and a populated `semantic_query`.
- The planner is one shared function used by every downstream consumer; no per-agent divergence is possible.
- Filter values are normalized (yes/no/na for flags; decoded labels for enums like status).
- All of the above is verified by unit tests with a stubbed LLM — no live LLM, index, or database required.
