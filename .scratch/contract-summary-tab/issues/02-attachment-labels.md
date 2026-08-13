# 02 — Humanized attachment labels in the detail view

**What to build:** every attachment row in the contract detail Attachments tab shows a bold plain-English label, the filename, and a human-readable size, e.g. **Signed contract** — Firebird ABC Distribution Contract - Feb. 6.pdf (2.3 MB). Mapping: signedcontract -> Signed contract, finalversioncontract -> Final version, DraftContract -> Draft, unspecified -> Other attachment; matching is case-insensitive; unknown values fall back to title-cased text; empty falls back to Other attachment.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] All four known field names render their mapped labels
- [x] Unknown and empty field names render sensible fallbacks
- [x] File sizes render human-readable (KB/MB)
- [x] Unit tests cover the mapping and fallback behavior
