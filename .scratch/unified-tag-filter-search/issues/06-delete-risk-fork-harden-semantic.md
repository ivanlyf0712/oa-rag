Status: ready-for-agent
Type: task

# 06 — Contract phase: delete the risk-search fork + harden the semantic path

**What to build:** once the unified tool, grounded generation, and the Streamlit UI no longer touch the legacy stack, delete the parallel risk-search path so the bug class it caused cannot regress. Remove the `risk_search` tool, its mode-based planner contract (`risky_contracts`/`general_search`), and the app/CLI exact-ref band-aids (the `_exact_contract_match`-style helpers and the `_search_exact_ref` shortcut) now superseded by the deterministic filter path. Independently, harden the hybrid semantic path so a missing graph extra (the GrandCypher `ImportError`) can no longer crash title or ref-in-sentence queries, since semantic remains the fallback for unrecognized identifiers.

**Blocked by:** 03 — Unified contract-search tool; 04 — Grounded answer generation; 05 — Streamlit evidence + risk_level.

## Acceptance criteria
- No `risk_search` tool, no mode-based (`risky_contracts`/`general_search`) planner contract, and no app/CLI exact-ref special-casing remain.
- Both agents expose a single contract-search path; there is exactly one route-decision function.
- A title query (e.g. `CKTEST080604`) and a ref-in-sentence query never crash, including when the graph extra (GrandCypher) is unavailable.
- The semantic fallback still returns relevant results for unrecognized identifiers.
- The full test suite passes after the deletions — the removal is behaviour-preserving for all supported queries.
