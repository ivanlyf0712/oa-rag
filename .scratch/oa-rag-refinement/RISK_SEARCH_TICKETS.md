# Risky Contract Search Tickets

## Goal
Add a natural-language risky-contract search mode to OA-RAG that auto-detects risk intent, applies deterministic filters over decoded fields, scores contracts by weighted risk signals, and surfaces the result set in the Streamlit UI.

## Ticket 1 — Define planner contract for risky-contract mode
**Type:** Design / API contract  
**Depends on:** None

### Scope
Define the structured output returned by the query planner.

### Deliverables
- Add a planner schema with:
  - `mode`: `risky_contracts | general_search | clarify`
  - `confidence`
  - `filters[]`
  - `explanation`
  - `clarification_question` (only for `clarify`)
- Add an allowlist of filterable risk fields for v1.
- Define allowed normalized values (`yes`, `no`, `na`).

### Acceptance criteria
- The planner contract is documented and explicit.
- Unknown fields are not permitted.
- The schema is compatible with the current decoded-field model in `core/db.py`.

---

## Ticket 2 — Implement LLM-based risk intent detection
**Type:** Backend / planner  
**Depends on:** Ticket 1

### Scope
Use the LLM to classify user queries as risky-contract searches, general searches, or clarification cases.

### Deliverables
- Add a planner step before search.
- Detect explicit risk language only.
- Return structured filters when confidence is sufficient.
- Return `clarify` when confidence is low.

### Acceptance criteria
- Queries like “show risky contracts” route to `risky_contracts`.
- Queries like “problem contracts” do not auto-route unless explicitly tied to risk/compliance language.
- Low-confidence cases return a clarification question instead of guessing.

---

## Ticket 3 — Add deterministic risk filter application
**Type:** Backend / retrieval  
**Depends on:** Ticket 1, Ticket 2

### Scope
Apply structured filters against decoded contract fields to produce the risky-contract candidate set.

### Deliverables
- Apply filter clauses against the DataFrame / contract records.
- Support exact matching on decoded `yes/no/na` values.
- Reject unknown fields safely.

### Acceptance criteria
- `IsRisksAccepted = no` correctly filters the result set.
- Multiple clauses combine deterministically.
- Filtering does not rely on semantic similarity.

---

## Ticket 4 — Implement backend risk scoring and severity tiers
**Type:** Backend / scoring  
**Depends on:** Ticket 3

### Scope
Compute a weighted risk score and severity tier for each contract.

### Deliverables
- Add configurable weights per risk field.
- Compute a total risk score.
- Map score to severity tiers (`low`, `medium`, `high`).
- Emit matched risk signals for explanation.

### Acceptance criteria
- `IsRisksAccepted = no` has the highest or near-highest weight.
- The score is deterministic and explainable.
- Severity is computed by backend rules, not by the LLM.

---

## Ticket 5 — Add gating and sort behavior for risky-contract results
**Type:** Backend / UX behavior  
**Depends on:** Ticket 4

### Scope
Use the risk score to gate results and sort the survivors.

### Deliverables
- Define the minimum score / severity threshold for display.
- Filter out low-risk noise by default.
- Sort remaining results by risk score descending.

### Acceptance criteria
- High-risk contracts appear before medium-risk contracts.
- Results below the threshold do not appear in the default risky-contract view.
- The gate can be tuned through configuration.

---

## Ticket 6 — Surface risk score, severity, and matched signals in Streamlit
**Type:** UI  
**Depends on:** Ticket 4, Ticket 5

### Scope
Update the Streamlit app to present risky-contract results in a trustable way.

### Deliverables
- Show risk severity.
- Show numeric score.
- Show matched risk signals.
- Show the planner explanation or a short backend explanation.
- Keep the current browse/search experience for general search.

### Acceptance criteria
- Flagged contracts display score + severity + matched signals.
- Users can understand why a contract was flagged.
- General search behavior remains intact.

---

## Ticket 7 — Add clarification UX for ambiguous risk queries
**Type:** UI / planner fallback  
**Depends on:** Ticket 2

### Scope
Ask the user for clarification when the planner cannot confidently determine risk intent.

### Deliverables
- Render the planner’s clarification question in Streamlit.
- Block automatic risk filtering when confidence is low.
- Let the user choose between risky-contract mode and general search.

### Acceptance criteria
- Ambiguous inputs do not silently enter risky mode.
- Users get a clear, short clarification prompt.

---

## Ticket 8 — Add tests for planner, filters, scoring, and UI behavior
**Type:** Testing  
**Depends on:** Tickets 1–7

### Scope
Validate the full risk-search flow end to end.

### Deliverables
- Unit tests for planner output shape and allowlist enforcement.
- Tests for decoded-field filter application.
- Tests for score / severity calculation.
- Smoke test for Streamlit rendering of risk metadata.

### Acceptance criteria
- Planner returns the expected structure.
- Risk filters match decoded yes/no/na values correctly.
- Score and severity are deterministic.
- UI renders the new risk fields without breaking existing search views.

---

## Suggested implementation order
1. Ticket 1 — planner contract
2. Ticket 2 — intent detection
3. Ticket 3 — deterministic filters
4. Ticket 4 — score and severity
5. Ticket 5 — gating and sort
6. Ticket 6 — UI exposure
7. Ticket 7 — clarification UX
8. Ticket 8 — tests
