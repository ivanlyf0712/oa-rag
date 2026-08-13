# INDEX

## Status
- OA contract index path: `apps/search_index`
- Current state: **built and persisted** (rebuilt 2026-08-06, raw-first / de-normalized).
- Build: 260 contracts from `formtable_main_385` → 268 chunks, 267s, BGE-M3 embeddings, graph disabled.
- Verified: decoded state lines (e.g. `Over5M: yes`) are embedded in searchable text; metadata tags carry `decoded_fields` and `contextual_fields`.

## Raw-first de-normalization (this round)
The pipeline was de-normalized so the index preserves source-of-truth values and decodes coded fields. Two new structures are attached to every normalized contract (`core/db.py`):

- `decoded_fields`: `{field: {raw, label}}` for every coded field present.
- `contextual_fields`: verbatim raw values for non-narrative contextual columns.
- `search_context`: rendered `Field: label` lines, appended to the indexed text so both BM25 and the vector encoder see decoded meaning.

### Coded fields decoded (0/1/2 → no/yes/na)
Defined in `BOOLEAN_CODE_FIELDS`. Each is emitted as `{raw, label}` in `decoded_fields` and as a `Field: label` line in the indexed text.

- Threshold / risk flags: `Over5M`, `Over100M`, `WithEndDate`, `Saved`, `Solely`, `IncludingExternalGuarantees`, `IsAuthoritySufficient`, `IsPreAuthoritySufficient`, `IsRisksAccepted`, `IsRenew`, `IsMC`, `iscontractfinancial`, `needapreliminaryreviewbygroupl`, `PreliminaryReviewFlag`, `preliminaryreviewflag2`
- Need-approval flags: `FlagNeedLegal`, `FlagNeedGFN`
- Related-party / data / capex flags: `IfRelatedToData`, `relatedtocapexpropertyleasingc`, `generalpurchaseandoverhk50k`, `unlimitedliabilitiesorliabilit`
- Prompt display flags (business meaning when shown): `PromptForOver5M`, `PromptForJustificationsUnder5M`, `PromptForRelatedToData`
- Documentation completeness: `allrelevantdocumentationhasbee`
- Acknowledgement: `ihaveread10points`
- Hide flags doubling as workflow state: `hideforcontractenddate`, `hidefornorenew`, `hideforcontractfinancial`, `hideforkeyrisks`, `hideforriskwarningsection`

### Free-text narrative fields in contract body (`_extract_contract_text`)
- Existing: `finalversioncontract`, `signedcontract`, `DraftContract`, `detailedbudgetapprovalrecord`, `description`, `KeyChanges`, `HistoricalVersions`, `PreviousSignedContracts`, `CounterpartyName_MultiLine`, `bufinancerole`
- Added (justification / reason narratives): `ReasonToSubmitLT5M`, `ReasonsNoEndDa`, `NoteOnContractAmount`, `reasonsfornotuploadingallthedo`
- Added (risk assessment prompt answers): `assessmentprompt1`–`assessmentprompt7`

### Contextual raw fields surfaced verbatim (`CONTEXTUAL_FIELDS` → `contextual_fields`)
- Identity / lifecycle: `requestId`, `RefNo`, `TitleReferenceNoOfContract`, `contractstartdate`, `contractenddate`
- Commercial terms: `ProductServices`, `ContractAmountHKD`, `NoteOnContractAmount`, `contracttype`, `contract_type`, `revenueyear1`, `revenuetotal`, `revenueprevious`, `gpyear1`, `gptotal`, `gpprevious`, `npatyear1`, `npattotal`, `npatprevious`, `roicyear1percent`, `roictotalpercent`, `roicpreviouspercent`
- Approval routing / sign-off: `Status`, `BusinessApprovalLevel`, `FinanceApprovalLevel`, `DetailFinanceApprovalLevel`, `MatrixFinanceApprovalLevel`, `SignoffLevel`, `DisplayLevel`, `BUApprovalGrade`, `BusinessSecurityLevel`, `BusinessApproverSecurityLevel`, `FinalBusiness`, `FinalBUFinance`, `FinalGroupFinance`, `FinalLegal`, `BusinessApprover`, `BUFinanceApprover`, `GroupFinanceApprover`, `LegalApprover`, `RiskEndorser`
- Ownership / routing: `contractowner`, `entitycontractowner`, `requestor`, `businessunit`, `BUName`, `Department`, `DCHSigningEntity`, `dchsigningentity1`, `requestedbusinessunit`
- Workflow state: `reviewtier`, `requested_date`, `requested_time`
- Audit: `modedatacreater`, `modedatacreatedate`, `modedatamodifydatetime`

## What was left out (deliberately) from the *indexed text*
These columns are **not** in the searchable/embedded text (they carry no standalone contract meaning for retrieval). Most remain metadata-only in `record["raw"]`; several are now surfaced as display metadata in `contextual_fields` (see the update note below):

- **Pure UI / show-hide controls**: `hidefordraft2`, `HideForMC`, `HideForDraft`, `HideForFinal`, `hideforbusinessl2`, `hideforbusinessl3`, `hideforfinancel2`, `hideforfinancel3` (display-only, not business state).
- **System / form engine internals**: `formmodeid`, `modedatacreatertype`, `modedatacreatetime`, `modedatamodifier`, `form_biz_id`, `MODEUUID`, `constactid`, `orgprocess`, `processId` (workflow plumbing, not contract content).
- **Redundant approval-grade duplicates**: `BUApprovalGradeL1`, `BusinessSecurityL1`, `BusinessApproverSecurityL1`, `BusinessPreApprovalLevel`, `l2businessapprover`, `l3businessapprover`, `entityfinanceheadlilist`, `l2financeapprover`, `l3financeapprover`, `level2businessapprover`, `level3businessapprover`, `level1financeapprover`, `level2financeapprover` (mirror the primary approver/level columns already included).
- **Approver identity references** (group routing IDs): `PreBusinessApprover`, `FinalBusinessApprover`, `BuHead`, `EntityFinanceHead`, `BuGroupFinanceHead`, `GroupCFO`, `ManagementCommittee`, `BUFinanceApproverL1`.
- **Delete/process lifecycle flag**: `isDeleteProcess` is kept on the record (`is_deleted`).

Note: everything above remains available in `record["raw"]` for auditability — it is only excluded from the *indexed match surface* and the curated `contextual_fields`.

### Update: always-NULL columns audited & "display all columns" added
A per-column scan of all 260 rows identified **39 columns that are NULL/empty in every row** (e.g. `requestId`, `ProductServices`, `Solely`, `IsMC`, `PromptForOver5M`, `hidefor*`, `BUName`, `DCHSigningEntity`, `LegalApprover`, `BUApprovalGrade`, `assessmentprompt2-6`, …).

- **Do NULL columns affect the vectors? No.** Empty values are skipped everywhere in `core/db.py` (`_pick_first`/`_pick_text`/`_build_decoded_fields`/`_build_contextual_fields`), so always-NULL columns never entered the indexed text, decoded_fields, or contextual_fields — confirmed against the index. Trimming them from the registries produced **byte-identical vectors** (no reindex required for correctness).
- **Registry trimmed anyway** for a cleaner config: the 11 always-empty boolean flags, 10 always-empty contextual fields, and `assessmentprompt2-6` were removed from `BOOLEAN_CODE_FIELDS`, `CONTEXTUAL_FIELDS`, and the narrative candidates (all still in `record["raw"]`).
- **Content-bearing fields that were previously left out** (financial `%` metrics `gpyear1percent`/`gptotalpercent`/`npatyear1percent`/…, approver IDs `l2businessapprover`/`BuHead`/`ManagementCommittee`/…, audit/system IDs `processId`/`MODEUUID`/`isDeleteProcess`/…) were **added to `CONTEXTUAL_FIELDS`** so they are surfaced as display metadata. These are metadata-only (not in the indexed text) → **vectors unchanged**, but the index was rebuilt so chunk metadata now carries all 81 contextual fields.
- **app.py now displays all columns**: `_load_sections` flattens *every* decoded + contextual field present in the data (113 columns total) instead of a hardcoded list, and the Browse grid has a "Show all columns" toggle to render the full set on demand.

## What is already working
- `apps/search.py` builds txtai embeddings from contract records.
- `apps/build_index.py` is the build entrypoint.
- `core/db.py` reads from MySQL and de-normalizes contract rows (raw-first).
- Search infrastructure: chunking, enrichment, hybrid retrieval, metadata tags.
- Decoded boolean/risk state is searchable by meaning (verified: "over 5M" query).
- `apps/app.py` (Streamlit UI) now surfaces decoded labels instead of the old `legal_approval`/`overruled` booleans:
  - Search results table shows `Over5M`, `Over100M`, `FlagNeedLegal`, `IsRisksAccepted`, `PreliminaryReviewFlag`, `IfRelatedToData` (yes/no/na) + `amount_label`.
  - Browse view has a "Risk / approval filters (decoded)" expander with yes/no/na multiselects, a "Show all columns" toggle that renders all content-bearing raw + decoded + contextual columns (113 total), and a per-record "Record detail" expander showing `decoded_fields` (raw → label) and `contextual_fields` (verbatim).
  - Search "Raw results" detail shows decoded_fields + contextual_fields side by side per hit.

### Update: risky-contract natural-language search (`apps/risk_search.py` + "Risk Search" view)
A new **"Risk Search"** Streamlit view turns a natural-language query into deterministic risk screening. The LLM is used **only** to detect risk intent + extract filters — filtering, scoring, severity, gating and ranking are deterministic backend rules over the decoded yes/no/na fields.

- **Planner (`RiskPlanner.plan`)** returns `{mode: risky_contracts|general_search|clarify, confidence, filters[], explanation, clarification_question}`. Explicit risk language ("risk not accepted", "legal review", "over 5m", …) is routed by a deterministic keyword fallback (no LLM call); anything ambiguous or low-confidence (`<0.6`) returns `clarify` and asks the user. LLM down → also `clarify` (safe default).
- **Filter allowlist (v1):** `IsRisksAccepted, FlagNeedLegal, FlagNeedGFN, Over5M, Over100M, IsAuthoritySufficient, PreliminaryReviewFlag, IfRelatedToData, allrelevantdocumentationhasbee, IsRenew, IncludingExternalGuarantees, iscontractfinancial` (values normalized to yes/no/na; unknown fields/values are rejected).
- **Weighted scoring (`score_risk`):** each matched signal adds points (e.g. `IsRisksAccepted=no +50`, `FlagNeedLegal=yes +20`, `IfRelatedToData=yes +15`, `Over5M/Over100M=yes +10`); rows get `risk_score`, `risk_severity` (high ≥50, medium 20–49, low 0–19), `matched_signals`, `risk_explanation`. Weights/tiers are configurable via `RISK_WEIGHTS_JSON` / `RISK_SEVERITY_JSON`.
- **Gate + sort (`gate_and_sort`):** hard-gate at the medium threshold (configurable via `RISK_GATE_MIN_SCORE`), then sort by score descending.
- **Verified on the rebuilt index:** "risk not accepted" → 10 high-severity contracts (50–100). Scoring all 260 contracts gives 10 high / 55 medium / 195 low; gating at ≥medium yields 65 contracts.
- Tests: `tests/test_risk_search.py` (19 tests: allowlist, filters, scoring, gating, planner routing, UI smoke).

#### Ticket 9 update — full risk-tag coverage + click-to-summarize
- **Allowlist expanded to 19 content-bearing tags** (grouped): outcome/authority (`IsRisksAccepted`, `IsAuthoritySufficient`), need-approval (`FlagNeedLegal`, `FlagNeedGFN`), value thresholds (`Over5M`, `Over100M`, `WithEndDate`, `Saved`, `IncludingExternalGuarantees`, `IsRenew`, `iscontractfinancial`), review (`PreliminaryReviewFlag`, `preliminaryreviewflag2`, `needapreliminaryreviewbygroupl`), related-party/data/capex (`IfRelatedToData`, `relatedtocapexpropertyleasingc`, `generalpurchaseandoverhk50k`, `unlimitedliabilitiesorliabilit`), and documentation (`allrelevantdocumentationhasbee`). *Note: `Solely`, `IsMC`, `IsPreAuthoritySufficient` and the `PromptFor*` flags are always-NULL in the source and absent from the index, so they cannot be filtered; the summary uses their substantive counterparts (`Over5M`, `IfRelatedToData`).*
- **Weights expanded** to all groups (e.g. `unlimitedliabilitiesorliabilit=yes +15`, `IfRelatedToData=yes +15`, `relatedtocapexpropertyleasingc=yes +10`, `generalpurchaseandoverhk50k=yes +10`, `IncludingExternalGuarantees=yes +10`, `allrelevantdocumentationhasbee=no +15`). New spread: 16 high / 85 medium / 159 low (101 gated at ≥medium).
- **Keyword routing table** (`_KEYWORD_FILTERS`) routes phrases like "unlimited liability", "capex / property leasing", "external guarantees", "over hk50k", "financial contract", "incomplete documentation" to the right field without an LLM call; the LLM planner prompt now names the full tag set.
- **Click-to-summarize UI:** the Risk Search view has an "Inspect a risky contract" selector → "Summarize contract" → a readable, grouped summary (`summarize_contract`) built from `build_contract_context` (metadata + per-group risk tags + matched signals + score/severity). The LLM is given **only** that explicit context and instructed not to invent facts; it falls back to a fully deterministic grouped summary when no LLM is reachable. The evidence JSON is shown in an expander.
- Tests: `tests/test_risk_search.py` now 28 tests (added: expanded allowlist, new-tag keyword routing, new-tag scoring, context grouping, deterministic summary, LLM-receives-only-context, summarize-click UI smoke).

## What still needs fine-tuning
- Minor: tune which `search_context` lines enter the match surface vs. metadata-only (currently all decoded lines are indexed).
- A small amount of validation on real reviewer queries to confirm the decoded labels improve precision.

## Recommended direction (done)
- Preserve raw SQL fields for auditability. ✅
- Add decoded state labels for coded fields. ✅
- Expand the indexed text/tags with decoded context. ✅
- Keep query-time search logic out of scope. ✅ (unchanged)

## Current expectation
- The POC index is **complete for this round**: raw-first extraction, coded-field decoding, and contextual-field enrichment are all in place and verified against the live DB.

