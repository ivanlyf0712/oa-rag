# OA-RAG Implementation Spec

## Problem Statement

The existing corpchat-rag system is designed for WeCom chat/message search and still contains Onyx Chat, chat analytics, and PostgreSQL-oriented assumptions. The goal is to turn it into an OA-RAG contract risk screening product that searches contract records in MySQL, supports contract metadata filtering, and presents the results in a contract-focused Streamlit dashboard.

A key constraint is that the real source table is , which is not the simplified contract schema originally assumed. The implementation must therefore map search and filter concepts onto the actual columns in that table rather than inventing fields like  or .

## Solution

Build a contract-oriented retrieval system that keeps the proven txtai search stack — hybrid keyword/semantic search, query expansion, RRF fusion, and reranking — while swapping the data source to MySQL and replacing the UI with contract search, contract browsing, and dashboard views.

Remove Onyx Chat completely. The UI should no longer embed or reference it, and no contract user flow should depend on that deprecated chat path.

## User Stories

1. As a contract reviewer, I want to search contract records by free text, so that I can find relevant agreements quickly.
2. As a legal reviewer, I want to filter by contract amount, so that I can focus on high-value agreements.
3. As a finance user, I want to filter by department or business unit, so that I can review only the contracts relevant to my area.
4. As an operations user, I want to search across long contract text, so that I can find clauses and obligations buried in the document.
5. As a reviewer, I want to inspect contract metadata alongside search results, so that I can understand context before opening the full record.
6. As an analyst, I want the system to rank results using hybrid retrieval, so that I get both precise keyword matches and conceptually similar contracts.
7. As an analyst, I want query expansion and reranking to remain available, so that the search quality stays strong for contract language.
8. As a platform user, I want the contract UI to be free of Onyx Chat, so that I only see contract-focused tools.
9. As a developer, I want the search index to be built from MySQL, so that the system uses the production contract source of truth.
10. As a developer, I want the contract schema mapping to be explicit, so that future work can extend filtering without guessing field names.
11. As a reviewer, I want dashboard metrics for contract counts and workflow states, so that I can see a quick risk overview.
12. As a reviewer, I want to browse contracts in a table, so that I can sort, scan, and export them.
13. As a product owner, I want the implementation to preserve the existing txtai core, so that the migration is lower risk and faster to validate.
14. As a maintainer, I want all legacy chat and Onyx references removed from the new OA application, so that there is no confusing product overlap.
15. As a maintainer, I want index-building and search commands to keep working from the CLI, so that the system remains scriptable.
16. As a tester, I want clear acceptance criteria for schema mapping, indexing, search, and UI, so that I can validate the migration end to end.
17. As a reviewer, I want filters to respect the real database values, so that the results match the source records accurately.
18. As a reviewer, I want the search results to surface the most relevant contract sections, so that I can inspect the material clauses first.
19. As a user, I want the app to load without chat-specific dependencies, so that the contract workflow starts cleanly.
20. As a user, I want the interface to expose the right contract metadata columns, so that I can trust the table view as a screening tool.

## Implementation Decisions

- The system will continue using txtai embeddings and the existing retrieval stack patterns, but the indexed corpus will be contracts from MySQL instead of chat messages.
- The source table is , and the implementation must map contract title, body text, amount, department, dates, ownership, approval, and risk-related fields from the actual schema.
- Because the real table does not expose the simplified fields originally assumed, the search filter API must use the actual schema mapping rather than hard-coded placeholder names.
- The indexing pipeline will read directly from MySQL and normalize row values into contract documents before chunking and enriching them.
- Graph expansion is not part of the contract MVP because the existing graph model is chat-relationship-specific and does not fit the contract workflow.
- Query expansion and reranking remain in scope because they are domain-agnostic and improve contract retrieval quality.
- The UI will be rewritten as a contract search and browsing experience with dashboard metrics, not as a chat product.
- Onyx Chat and all embedded chat behavior will be removed from the OA application.
- The MySQL connection will use .
- The implementation should favor explicit, inspectable mapping helpers so the schema interpretation is visible and easy to adjust.

## Testing Decisions

- Validate the schema mapping by checking that the MySQL fetch helper returns records with the expected normalized keys.
- Validate the index build path by building an index from the real table and confirming that document chunks are created.
- Validate search behavior by exercising keyword, semantic, and hybrid retrieval against contract text.
- Validate filter behavior by verifying that selected contract metadata constraints remove non-matching records.
- Validate the UI by loading the Streamlit app and confirming that it renders contract search, contract browsing, and dashboard views without Onyx Chat.
- Existing prior art in the codebase includes CLI-driven search validation and Streamlit-driven UI smoke checks.

## Out of Scope

- Reintroducing any chat analytics or Onyx Chat functionality.
- Rebuilding the conversation graph model for contracts.
- Designing a new contract approval workflow or changing the underlying MySQL schema.
- Perfect semantic normalization of every possible contract column before the schema mapping is clarified.
- Full production-grade export/reporting workflows beyond the initial browse and dashboard views.

## Further Notes

The migration should stay close to the existing retrieval architecture and isolate the schema adaptation in a small number of modules. The biggest unknown is how the real  columns should map into user-facing contract concepts, so the first implementation step must be schema-driven rather than assumption-driven.
