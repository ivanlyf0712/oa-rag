# 04 — View Detail tabs rebuild (Raw / Contextual / Risk)

**Spec:** `docs/specs/unified_search_risk_merge.md`

**What to build:** the contract detail view becomes three lenses over one stored record. **Raw** (replaces Decoded): the complete `raw` record (all ~155 columns) as JSON, no curation. **Contextual**: only self-explanatory fields — plain-value fields (identity, dates, amounts, names, workflow state) plus the 20 coded fields rendered as decoded labels ("Yes/No"), grouped per `data/contract-data-column-groups.md`; audit/system identifiers and undecodable codes excluded (they remain in Raw); missing values render an explicit "(empty)" placeholder. **Risk**: score, severity, matched signals with human-readable labels, derived from the same normalized record backing Contextual (single source of truth). No index rebuild — `raw` is already in the index metadata. This fixes the "fields sometimes not showing" symptom at the rendering layer.

**Blocked by:** None — independent of 01–03 (touch only the detail renderer), but lands cleanly alongside 03.

**Status:** done

- [x] Raw tab renders the complete stored record as JSON (every column, no omissions)
- [x] Contextual tab: curated self-explanatory fields + decoded coded-field labels, grouped, with "(empty)" placeholders for missing values
- [x] Risk tab: score/severity/labeled signals derived from the same normalized record as Contextual
- [x] Sparse contracts render fully (placeholders) instead of partial/empty JSON
- [x] Opaque IDs / undecodable codes absent from Contextual but present in Raw
- [x] Placeholder-filling and curation helpers are pure and unit-tested; tabs manually verified on a sparse and a rich contract
