Status: ready-for-agent
Type: task

# 07 — Acceptance gate: the four agreed queries reproducible against the index

**What to build:** an end-to-end check proving the unified pipeline against the real index (`apps/search_index`), covering the four acceptance queries agreed in the spec. This is the bar that demonstrates the feature works for the user's bottom line — the correct contract is recognized and used every time.

**Blocked by:** 06 — Delete the risk fork + harden the semantic path.

## Acceptance criteria
- `CCA20250096` → exactly `contract_109` with full metadata.
- "Is contract CCA20250096 high risk?" → resolves to contract 109; the verdict comes from its tags; attachment facts are cited.
- "Find completed contracts with high risk" → filters `status=completed` plus the high-risk threshold, sorted by score, with no semantic drift; each row shows `risk_level`.
- "Contracts mentioning unlimited liability" → the hybrid semantic path returns relevant clause passages.
- All four pass reproducibly against `apps/search_index` (268 sections, 259 distinct `ref_no`).
