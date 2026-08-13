Status: ready-for-agent
Type: task

# 05 — Streamlit: evidence panel + `risk_level` column via the unified path

**What to build:** the Streamlit app consumes the unified tool (ticket 03) for its evidence re-run instead of any app-level exact-ref special-casing. Evidence rows are normalized to the `{id, text, score, metadata}` shape so decoded and contextual fields render fully populated, fixing the bare `contract_109__chunk0` with `score None` and empty metadata symptom. The results table always includes a `risk_level` column for every row, regardless of how the query was phrased.

**Blocked by:** 03 — One unified contract-search tool.

## Acceptance criteria
- The evidence re-run uses the unified filter path; there is no app-level exact-ref helper.
- Evidence rows render with decoded and contextual fields fully populated (no bare id with `score None`).
- The results table always shows a `risk_level` column for every result row.
- Verified with a fake `st` render harness — no live Streamlit server, LLM, or index required.
