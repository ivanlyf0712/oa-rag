# 02 — Contract index build pipeline

**What to build:** a contract-oriented txtai index build flow that reads normalized records from MySQL, turns them into searchable contract documents, and persists an index that the OA search stack can load without any PostgreSQL or chat-specific assumptions.

**Blocked by:** 01 — Schema mapping and MySQL contract fetch

**Status:** done (validated live against MySQL `oa_rag.formtable_main_385`)

## Detailed change plan

1. Replace the OA build path with a contract-first indexer.
   - Remove the remaining chat/message build assumptions from the OA index pipeline.
   - Read normalized records through the shared contract fetch helper.
   - Keep txtai as the persistence and retrieval engine.

2. Define a contract document model for indexing.
   - Create one searchable document per contract row or row chunk, depending on text length.
   - Build the indexed text from the contract fields that matter for retrieval, such as title, counterparty, department, amount, status, dates, key flags, and the best available contract text field.
   - Preserve a stable source identifier so search results can be traced back to the original MySQL row.

3. Preserve hybrid search compatibility.
   - Ensure indexed content supports both keyword and semantic retrieval.
   - Keep the existing txtai hybrid index shape and scoring behavior available to the later search ticket.
   - Do not reintroduce the chat graph relationship model into the contract build path.

4. Persist and reload the index cleanly.
   - Save the built txtai index under the OA app’s index location.
   - Make the build command idempotent so rebuilding the same source data produces a usable contract index.
   - Keep the build entrypoint callable from the CLI.

5. Make the contract metadata available for later filtering and UI display.
   - Carry forward only the normalized fields that improve retrieval, filtering, and result display.
   - Prefer stable identity, searchable facets, workflow/risk flags, and date/amount context over internal bookkeeping fields.
   - Do not encode any UI-specific behavior here; only supply the indexed data shape.

6. Validate with a real MySQL build.
   - Build the index from the live contract table.
   - Confirm the output contains contract records rather than chat messages.
   - Confirm the index can be loaded by the OA search layer in the next ticket.

**Acceptance criteria:**
- [x] Running the OA build command creates a txtai index from MySQL contract rows. (260 contracts → 261 chunks, saved+reloaded)
- [x] Indexed documents contain contract text plus the metadata required for later search/filter UI work. (`sections.tags` carries contract_id, ref_no, title, counterparty, department, amount, dates, status, contract_type, legal_approval, overruled)
- [x] The build flow no longer depends on PostgreSQL, messages, contacts, or other chat-only tables. (psycopg2/DB_CONFIG/messages/contacts removed; reads via `core.db.fetch_contracts()`)
- [x] The generated index can be opened by the OA search stack without conversion. (loaded via `load_index`, hybrid search verified)
- [x] A live build against `formtable_main_385` completes successfully. (graph-off 108s / graph-auto 112s, `--graph-expand` returns `same_contract_type` hits)
