# 03 — Summary tab (first tab, button-gated, cached)

**What to build:** the contract detail view gains a Summary tab in first position. Nothing is generated until the reviewer presses Generate summary (with a caption noting it uses the LLM). Generation shows a spinner, then renders two headed sections — Document summary (from the signed attachment) and Risk assessment (from recorded risk tags) — a caption naming the source file and its humanized label, and an explicit notice when degraded (no readable attachment / extraction failed / LLM unavailable). Results are cached process-wide keyed by (ref_no, attachment mtime); revisits are instant; a Regenerate button forces a fresh call. The existing summarize_contract_with_attachment backend seam and the existing _get_summary_llm helper are reused unchanged.

**Blocked by:** 01 — Resolve attachment paths against UPLOADS_ROOT; 02 — Humanized attachment labels (the source-file caption reuses the label helper).

**Status:** done

- [x] Summary is the first tab; existing four tabs keep their order after it
- [x] No LLM call happens until Generate summary is pressed
- [x] Two sections render with source-file caption and degraded-mode notices
- [x] Repeat views are served from cache without an LLM call
- [x] Regenerate bypasses the cache; an attachment file change (mtime) invalidates automatically
- [x] Backend seam tests extended for cache hit/miss and mtime invalidation
