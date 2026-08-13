# Spec: Contract Detail — Attachment Labels + Summary Tab

> Status: ready-for-agent
> Builds on: the already-implemented (but unwired) `apps/attachment_summary.py` backend and its 13 tests, `risk_search.build_contract_context` / `score_risk`, and the unified UI from `docs/specs/unified_tag_filter_search.md`.
> Verified facts: `contract_attachments.field_name` in {signedcontract x3, finalversioncontract x2, DraftContract x3, unspecified x1}; attachments tab currently renders bare filenames only; `_get_summary_llm()` already exists in the app; in the Docker container, uploads live at /app/uploads while DB file_path values are host-absolute.

## Problem Statement

A contract reviewer opening the per-contract detail view hits three problems:

1. **Attachments are anonymous.** The Attachments tab lists bare filenames (e.g. `Firebird ABC Distribution Contract - Feb. 6.pdf`) with no indication of *which kind* of document each file is. The OA system records this — `field_name` distinguishes signedcontract / finalversioncontract / DraftContract — but it is not displayed, so the reviewer cannot tell the signed contract from a draft without opening each file.
2. **No at-a-glance review summary.** Understanding a contract means reading the Decoded fields, Contextual fields and Risk fields tabs plus opening the attachment files manually. The reviewer wants one generated summary that combines (a) the recorded risk tags, (b) the deterministic risk score/severity, and (c) the actual content of the signed contract attachment — but must not pay LLM tokens when they are only clicking through contracts quickly.
3. **Latent Docker breakage for attachment reads.** The database stores host-absolute file paths (`/home/.../oa-rag/uploads/...`), while the dockerized app mounts uploads at /app/uploads. Any feature that reads attachment bytes silently degrades in the container unless paths are resolved against a configurable root.

## Solution

- **Label every attachment row** with a humanized form of its `field_name`: **Signed contract** / **Final version** / **Draft** / **Other attachment**, rendered as a bold prefix on the filename with a human-readable file size.
- **Add a Summary tab as the first tab** of the contract detail view. Nothing is generated until the reviewer presses **Generate summary**; generation produces two headed sections — *Document summary (from the signed attachment)* and *Risk assessment (from recorded risk tags)* — plus a caption naming the source file and its label, and an explicit notice line whenever a degradation occurred (no readable attachment / extraction failed / LLM unavailable).
- **Cache generated summaries process-wide**, keyed by (ref_no, attachment file mtime), so revisiting a contract is instant and a re-synced file automatically invalidates its summary. A **Regenerate** button bypasses the cache on demand.
- **Resolve attachment paths at read time** against an `UPLOADS_ROOT` environment setting (repo uploads/ locally, /app/uploads in the container), leaving stored DB values untouched.

## User Stories

1. As a contract reviewer, I want each attachment row to show whether it is the signed contract, the final version, or a draft, so that I know which file is authoritative without opening it.
2. As a contract reviewer, I want labels in plain English (not raw OA field keys like signedcontract), so that the UI reads naturally.
3. As a contract reviewer, I want unrecognised or unspecified field names to still display sensibly, so that no attachment row looks broken.
4. As a contract reviewer, I want each attachment'+chr(39)+'s file size shown, so that I can gauge how large a document is before opening it.
5. As a contract reviewer, I want a Summary tab as the first tab of the detail view, so that the highest-value overview is what I land on.
6. As a contract reviewer, I want summary generation to happen only when I press a button, so that quick navigation between contracts never spends LLM tokens.
7. As a contract reviewer, I want a spinner while the summary generates, so that I know work is in progress.
8. As a legal reviewer, I want the document summary to come from the signed contract attachment (falling back to final version, then draft), so that the summary reflects the binding text whenever it exists.
9. As a legal reviewer, I want the document summary to cover parties, scope, value/term and notable obligations or unusual clauses in one concise paragraph, so that I get the essence without reading the file.
10. As a compliance officer, I want the risk assessment section to be built from the recorded risk tags plus the deterministic risk score and severity, so that the summary is consistent with the Risk fields tab.
11. As a compliance officer, I want a clear notice when the summary had to fall back (missing/unreadable attachment, LLM down), so that I never mistake a degraded summary for a full one.
12. As a contract reviewer, I want the summary to name the exact file it was generated from, so that I can audit the source.
13. As a contract reviewer, I want re-opening a contract'+chr(39)+'s summary to be instant, so that flipping between contracts costs no extra tokens.
14. As a contract reviewer, I want the cached summary to invalidate automatically when the attachment file changes, so that I never read a stale summary.
15. As a contract reviewer, I want a Regenerate button, so that I can force a fresh take when I know something changed.
16. As an operator, I want attachment paths to resolve correctly both on the host and inside Docker without migrating the database, so that the feature works identically in both deployments.
17. As an operator, I want the summary to reuse the existing LiteLLM proxy model (dseek-v4-flash), so that no new provider configuration is needed.
18. As a reviewer, I want summaries written in English, matching the contract documents.

## Implementation Decisions

- **Labels**: a single field-name -> label mapping (signedcontract -> Signed contract, finalversioncontract -> Final version, DraftContract -> Draft, unspecified -> Other attachment) with a title-cased fallback for unknown values, exposed as a small pure helper alongside the existing attachment module, and used by the Attachments tab renderer. Matching is case-insensitive on the stored value.
- **Summary tab**: the detail view'+chr(39)+'s tab list becomes Summary, Decoded fields, Contextual fields, Risk fields, Attachments. The Summary tab shows the Generate button; after first generation it shows the cached result with a Regenerate button beside it.
- **Backend reuse, unchanged interface**: the tab calls the existing summarize_contract_with_attachment(row, llm, cache) seam. Its two-section output, its signed -> final -> draft attachment priority, its 10k-character extraction cap, and its fallback notices are all adopted as-is.
- **LLM**: the existing _get_summary_llm() helper (session-cached, LiteLLM proxy, dseek-v4-flash, graceful None) is injected. No new provider configuration; a future model override can be added via environment without code changes.
- **Caching**: a module-level (process-wide) dictionary keyed by (ref_no, attachment mtime) is passed as the backend'+chr(39)+'s cache. Regenerate deletes the key before calling the backend.
- **Path resolution**: attachment file paths are resolved at read time — if the stored path is not present on disk, the portion from the uploads/ marker onward is re-anchored under UPLOADS_ROOT (environment variable; default: the repo'+chr(39)+'s uploads/ directory; the compose app service sets it to /app/uploads). No database migration; sync_attachments.py keeps writing what it writes today.
- **No schema changes, no changes to risk scoring, no changes to the search stack.**

## Testing Decisions

- **Seam**: the existing, highest-level seam — the public functions of the attachment-summary module (list_attachments, choose_attachment, summarize_contract_with_attachment) with their injected db_connect / llm / cache collaborators. No new seams are introduced; the Streamlit tab wiring is kept thin enough to stay untested (consistent with the repo, which has no Streamlit UI test framework).
- **Prior art**: tests/test_attachment_summary.py (13 tests) already exercises extraction, priority, fallbacks and caching with fakes; the new tests extend that file.
- **New tests**: (a) label mapping for all four known field names plus unknown/empty fallbacks; (b) path remap — host-absolute stored path resolves under a temporary UPLOADS_ROOT; missing files still yield the no-readable-attachment notice; (c) cache invalidation on mtime change.
- Tests assert external behavior only (returned labels, chosen attachment, notice strings, cache hits), never implementation details.

## Out of Scope

- OCR of scanned/image-only attachments.
- Attachment download links or in-app file preview.
- Rewriting stored file_path values or any other database migration.
- A per-feature model override (e.g. SUMMARY_LITELLM_MODEL) — deferred until flash-class quality proves insufficient.
- Multi-user/auth concerns (single-operator tool).

## Further Notes

- After the change lands, the Docker image must be rebuilt (only the COPY layers re-run) and the app container recreated; verify on CCA20260156 (has all three attachment kinds) and CCA20250096.
- Remembered lesson: the container runs Python 3.12 (eager annotations) while the local venv is 3.14 (lazy annotations) — never let a name used in an annotation be defined only inside a function.
