# Risky Contract Search Planner Spec

## Goal
Support a natural-language search mode focused on identifying **risky contracts** in the OA contract screening app.

The primary use case is not semantic retrieval. It is:
- detect when the user wants risky contracts,
- map that request to deterministic filters,
- score and rank contracts by risk,
- explain why each result was flagged.

## Scope
This spec applies to the Streamlit app in `apps/oa/app.py` and the contract normalization / indexing pipeline in `core/db.py` and `apps/oa/search.py`.

It introduces a **planner step** in front of search.

## Core design principle
Use the LLM for **intent detection and filter extraction** only.

Use backend rules for:
- filtering,
- scoring,
- severity assignment,
- result ordering.

The LLM must **not** decide whether a contract is truly risky.

## User experience
When a user asks a clear risk-related question such as:
- “show risky contracts”
- “find contracts with unaccepted risk”
- “which contracts need legal review?”

the system should automatically switch into **risky contract mode** and apply risk filters.

When the query is ambiguous, the app should ask a clarification question instead of guessing.

## Planner output contract
The planner should return structured data in this shape:

```json
{
  "mode": "risky_contracts | general_search | clarify",
  "confidence": 0.0,
  "filters": [
    {
      "field": "IsRisksAccepted",
      "op": "=",
      "value": "no"
    }
  ],
  "explanation": "Short natural-language reason for the chosen mode",
  "clarification_question": "Optional question when mode=clarify"
}
```

### Fields
- `mode`
  - `risky_contracts`: use deterministic risk filters and scoring.
  - `general_search`: use the existing normal search pipeline.
  - `clarify`: ask the user to disambiguate.
- `confidence`
  - Planner confidence in the mode / filter extraction.
- `filters`
  - Structured clauses from an allowlist only.
- `explanation`
  - Short human-readable justification for logging and debug UI.
- `clarification_question`
  - Required when `mode=clarify`.

## Allowed filter vocabulary for v1
Keep the planner constrained to the current fixed risk-tag set used by OA-RAG.

### Core risk filters
The LLM may emit only these fields in v1:

#### Threshold / risk flags
- `Over5M`
- `Over100M`
- `WithEndDate`
- `Saved`
- `Solely`
- `IncludingExternalGuarantees`
- `IsAuthoritySufficient`
- `IsPreAuthoritySufficient`
- `IsRisksAccepted`
- `IsRenew`
- `IsMC`
- `iscontractfinancial`
- `needapreliminaryreviewbygroupl`
- `PreliminaryReviewFlag`
- `preliminaryreviewflag2`

#### Need-approval flags
- `FlagNeedLegal`
- `FlagNeedGFN`

#### Related-party / data / capex flags
- `IfRelatedToData`
- `relatedtocapexpropertyleasingc`
- `generalpurchaseandoverhk50k`
- `unlimitedliabilitiesorliabilit`

#### Prompt display flags
- `PromptForOver5M`
- `PromptForJustificationsUnder5M`
- `PromptForRelatedToData`

#### Documentation completeness
- `allrelevantdocumentationhasbee`

### Supported values
All filter values should normalize to one of:
- `yes`
- `no`
- `na`

The backend must reject unknown fields and normalize unexpected values conservatively.

### Planner behavior over the full tag set
The planner should consider **all risk-related tags** in the allowlist when extracting filters and explaining intent. For v1, the planner still remains conservative: it may emit only the fields above, but it should recognize the whole set as relevant risk evidence and mention them in explanations / summaries when present.

## Risk mode trigger policy
Risk mode should trigger only on **explicit risk language**.

### Examples that should trigger risk mode
- risky
- risk
- risk accepted
- unaccepted risk
- legal review
- legal approval
- authority insufficient
- over 5M
- related to data
- preliminary review
- documentation incomplete

### Examples that should not automatically trigger risk mode
- problem
- issue
- concern
- bad

These vague terms are too broad for v1.

## Fallback behavior
If the planner has low confidence or cannot confidently extract risk filters:

1. Prefer `mode=clarify`.
2. Ask the user a short clarification question.
3. Do **not** silently guess.

Example clarification:
> “Do you want me to search for risky contracts, or perform a general contract search?”

## Risk scoring model
The backend should compute risk score using a **weighted rules system**.

### Recommended scoring behavior
- Each matched risk signal contributes points.
- Strong signals should have higher weights.
- Weights should be configurable and versioned.
- The total score determines severity.

### Example score rule set
- `IsRisksAccepted = no` → 50 points
- `FlagNeedLegal = yes` → 20 points
- `FlagNeedGFN = yes` → 15 points
- `Over5M = yes` → 10 points
- `IsAuthoritySufficient = no` → 20 points
- `PreliminaryReviewFlag = yes` → 10 points
- `IfRelatedToData = yes` → 15 points
- `allrelevantdocumentationhasbee = no` → 10 points

### Severity thresholds
Severity should be computed by backend rules only.

Recommended tiers:
- `low`
- `medium`
- `high`

Example threshold model:
- `0–19` → low
- `20–49` → medium
- `50+` → high

## Gate and sort policy
The risky-contract view should **hard-gate first, then sort within each bucket**.

Default behavior:
1. Filter to contracts meeting the minimum risk threshold.
2. Sort by risk score descending.
3. Show the highest-risk contracts first.

This reduces noise while keeping priorities visible.

## Result explanation requirements
For each flagged contract, the UI should show:
- severity tier
- numeric score
- matched signals
- short explanation

Example display:

> **High risk — 82**  
> Matched: risk accepted = no, legal review = yes, over 5M = yes  
> Reason: The contract includes explicit unaccepted risk and requires legal review.

## UI requirements
The Streamlit app should support two modes:

### Normal search
Keep existing search behavior for general contract discovery.

### Risky contract search
When the planner selects `risky_contracts`:
- apply deterministic filters,
- compute risk score,
- show risk severity,
- surface matched signals and explanation.

### Contract summary on click
When a user clicks a risky contract row, the app should display a readable contract summary panel with:
- contract identifiers and basic metadata,
- a plain-language explanation of the risk signals,
- related-party / data / capex context from:
  - `IfRelatedToData`
  - `relatedtocapexpropertyleasingc`
  - `generalpurchaseandoverhk50k`
  - `unlimitedliabilitiesorliabilit`
- prompt/business-meaning flags from:
  - `PromptForOver5M`
  - `PromptForJustificationsUnder5M`
  - `PromptForRelatedToData`
- other relevant high-signal tags when present.

The summary may be generated by a backend LLM, but the underlying evidence should be shown explicitly and the LLM should only summarize the provided context.

## Search pipeline integration
Recommended flow:

1. User enters query.
2. Planner classifies intent.
3. If `mode=clarify`, ask a question.
4. If `mode=risky_contracts`:
   - apply structured filters to the decoded fields,
   - compute weighted risk score,
   - gate by severity threshold,
   - sort by score descending,
   - render matched signals and explanation.
5. Otherwise fall back to the current general search pipeline.

## Data dependencies
This spec assumes the current de-normalized fields already exist:
- decoded fields in `decoded_fields`
- contextual fields in `contextual_fields`
- searchable context lines in `search_context`

The risk search planner should primarily operate on the decoded fields, not semantic text.

## Non-goals
- No learned risk model in v1.
- No open-ended LLM-generated SQL.
- No semantic retrieval as the primary mechanism for risk detection.
- No broad vague-issue auto-routing into risk mode.

## Acceptance criteria
This feature is done when:
- Natural-language risk queries are auto-detected reliably.
- The planner returns a structured mode + filters payload.
- Risk mode uses deterministic filters, not semantic similarity.
- Risk score and severity are backend-controlled.
- Low-confidence queries produce a clarification question.
- Results show score, severity, matched signals, and explanation.
