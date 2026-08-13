# 03 — Contract search and metadata filtering

**What to build:** a contract search experience that supports keyword, semantic, and hybrid retrieval plus metadata-based filtering using the actual contract fields from the MySQL schema.

**Blocked by:** 01 — Schema mapping and MySQL contract fetch; 02 — Contract index build pipeline

**Status:** ready-for-agent

## Current code state

- The build pipeline already indexes contract rows from MySQL into txtai.
- Each indexed chunk currently carries `sections.tags` metadata with:
  - `contract_id`
  - `request_id`
  - `ref_no`
  - `title`
  - `counterparty_name`
  - `product_services`
  - `department`
  - `department_id`
  - `amount`
  - `amount_label`
  - `contract_start_date`
  - `contract_end_date`
  - `requested_date`
  - `status`
  - `contract_type`
  - `legal_approval`
  - `overruled`
  - `chunk_index`
- Search already loads the txtai index and supports keyword / semantic / hybrid / expanded search.
- Search filtering is now aligned to contract metadata instead of chat fields:
  - `contract_type` is exposed via the CLI as `--contract-type` (internally still passed as the filter value)
  - `--contract-type` supports comma-separated values
  - date filters use `requested_date`
  - graph expansion works with contract structural relations (`same_contract`, `same_counterparty`, `same_department`, `same_contract_type`)
- Result rendering is contract-first:
  - table headers now show `Counterparty` and `Contract Type`
  - synthetic benchmark now validates `contract_type` and `counterparty_name` instead of chat labels

## Detailed operation plan

1. Align search filters to contract semantics.
   - Treat contract metadata as the source of truth for all filtering.
   - Replace any remaining chat-oriented filter assumptions with contract fields.
   - Ensure filtering happens on the metadata attached to each retrieved chunk, not on raw text.

2. Define the filterable contract facets.
   - Primary facets: `counterparty_name`, `department`, `contract_type`, `status`.
   - Secondary facets: `legal_approval`, `overruled`.
   - Context filters: `requested_date`, `contract_start_date`, `contract_end_date`, `amount`.
   - Identity/debug facets: `contract_id`, `ref_no`, `chunk_index`, `request_id`.

3. Preserve the retrieval stack.
   - Keep txtai hybrid search as the retrieval engine.
   - Keep query expansion and reranking working over contract chunks.
   - Maintain the existing result ranking pipeline so this ticket only changes contract semantics, not search architecture.

4. Surface contract metadata in results.
   - Display user-facing fields like title, counterparty, department, amount, dates, and status.
   - Keep internal fields out of the visible UI unless they help debugging or traceability.
   - If needed later, add human-readable mapping for `contract_type` codes instead of exposing raw IDs.

5. Keep graph expansion contract-aware.
   - Use the contract structural relations already built by the indexer.
   - Avoid reviving chat relationships or chat-specific graph semantics.
   - Ensure graph traversal only expands over contract-relevant neighbors.

6. Validate against the live index.
   - Run keyword, semantic, hybrid, and expanded searches against the live OA contract index.
   - Confirm filters return the correct contract chunks.
   - Confirm result cards show contract metadata instead of chat metadata.

## Acceptance criteria

- [x] Search results can be filtered by contract metadata derived from the real schema.
- [x] Hybrid retrieval, query expansion, and reranking continue to work over contract content.
- [x] The search layer uses `contract_type`, `requested_date`, `counterparty_name`, `department`, and related contract fields rather than chat labels or message timestamps.
- [x] Graph expansion continues to work with the contract structural relations emitted by the build pipeline.
- [x] The search output shows contract-oriented metadata in result cards and tables.
- [x] The CLI exposes contract-first filtering via `--contract-type`, including comma-separated values.
