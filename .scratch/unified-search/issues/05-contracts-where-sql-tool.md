# 05 — (Phase 2) `contracts_where` SQL tool

**Spec:** `docs/specs/unified_search_risk_merge.md` (Phase 2 section)

**What to build:** a second ReAct tool ported from corpchat's `search_messages_where`: `contracts_where(condition)` takes a natural-language condition ("contracts over HK$5M ending before 2027", "contracts needing legal review"), translates it to validated SQL over the sections SQLite DB (rule-based `_condition_to_sql` first, LLM translation as fallback, `_validate_sql` guardrails, index-scan last resort), and returns matching contracts as the same `ContractRow` set into the same result store — one table, one detail view regardless of which tool ran. This justifies the retained ReAct loop: the agent now genuinely chooses between semantic search and exact structured retrieval.

**Blocked by:** 02 — LLM observation formatter, result store & agent wiring (needs the unified row type + result store).

**Status:** done

- [x] `contracts_where` registered as a second tool alongside `search_contracts`
- [x] Rule-based condition→SQL covers the common shapes (amount comparisons, date bounds, coded-flag labels)
- [x] LLM translation fallback with SQL validation (read-only, single statement, sections table only)
- [x] Index-scan fallback when translation fails; never an unhandled exception to the agent
- [x] Results land in the result store as `ContractRow`s; table/detail rendering identical to semantic path
- [x] Tests ported in spirit from corpchat `test_tools_expansion.py`: rule path, LLM path (mocked), invalid SQL rejected, fallback path
- [x] Primary use case works: "list all contracts with <condition>" → WHERE clause → grounded list
- [x] Bare "list all" / empty condition → no WHERE clause → returns every contract (subject to the standard 50-row LLM budget + overflow marker; table shows all)
- [x] Planner/routing rule documented: "list all"-style queries with no semantic content route to the structured path (this tool or the service directly), never to vector search
