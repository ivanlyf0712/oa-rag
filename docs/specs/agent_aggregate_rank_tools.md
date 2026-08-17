# Agent Aggregate & Rank Tools

**Spec status:** ready-for-agent
**Source:** grilling session 2026-08-17; all decisions below were explicitly confirmed by the user.
**Depends on:** `docs/specs/fast_route_comparative_gate.md` (analytical risk queries now escalate to the ReAct engine; this spec gives that engine something to *compute* with).

## Problem Statement

The ReAct LLM today has only two tools, and both merely **fetch rows**:

- `contract_search` → semantic (hybrid keyword/vector) rows.
- `contracts_where` → SQL-filtered rows.

Neither can **aggregate, compare, rank, or drill in**. The fast-route comparative gate (prior spec) now correctly escalates analytical queries — "compare risk-not-accepted contracts across departments", "how many by month", "which department has the most", "total value" — to the ReAct engine. But once there, the LLM can only retrieve a pile of rows and eyeball a prose answer over them. That is the ceiling on the agent being "smart": the routing is now principled, but the engine has nothing to *reason with*.

The SQL backend (`apps/search/service.py::search_where`, `apps/search/where_sql.py`) already supports arbitrary validated `WHERE` over the `sections` table (whose `tags` JSON holds `amount`, `department`, `counterparty_name`, `contract_type`, `status_label`, `contract_start_date`, `contract_end_date`). It is only ever used to *filter*, never to *aggregate*.

## Solution

Give the ReAct engine two new capabilities, both whitelist-driven and built on the existing validated-SQL safety layer:

1. **`contracts_aggregate` tool** — SQL-side aggregation with structured, whitelisted params (no free SQL). The LLM supplies a metric, an optional group-by, and an optional natural-language condition; the code builds and validates the query.
2. **`rank_by` param on `contract_search`** — whitelisted sort (`relevance` / `risk` / `amount`) applied to the candidate set before formatting, enabling "top N by amount / by risk".

Decisions locked in the grilling session:

1. **Scope = `contracts_aggregate` + `rank_by` first.** `contract_detail` and `contracts_compare` are deferred (see Out of Scope). Aggregation is the highest-leverage unlock; compare can be emulated by two aggregate calls.
2. **SQL-side aggregation via templated, whitelisted params** — not fetch-then-aggregate in Python. Fetching a `LIMIT`-capped row set and then summing it in Python yields *wrong totals presented as fact*; the DB must do the aggregate over the full matching set.
3. **Structured signature, no free SQL** — the LLM picks from enums; the only free text is `condition`, which is already validated by `condition_to_sql`/`_validate_sql`. The failure mode becomes "wrong enum" (recoverable) rather than "bad SQL" (silent wrong answer).
4. **Output = compact rendered table + one-line summary**, bounded to ~50 groups with an explicit "+N more" overflow marker, matching the existing unified-search observation style. The LLM reads it directly; the UI already renders observations as evidence — no new UI work.
5. **Defer everything else** — `contract_detail`, `contracts_compare`, extra metrics (`min`/`max`/median), extra group-bys, and any UI charting are all out of scope this increment.

## User Stories

1. As a contracts reviewer, I want "how many contracts per department" answered with an actual count, so that I get a number, not a narrative guess over a sample.
2. As a contracts reviewer, I want "total contract value by department" computed over *all* matching rows, so that the figure is correct rather than truncated by a retrieval cap.
3. As a risk officer, I want "compare risk-not-accepted contracts across departments" to return a per-department breakdown, so that I can see the distribution rather than a flat list.
4. As a contracts reviewer, I want "which counterparty has the highest total contract value" answered by a ranked aggregate, so that the superlative is computed, not eyeballed.
5. As a contracts reviewer, I want "show the top 5 contracts by amount" to return them sorted by amount, so that ranking reflects my intent rather than semantic relevance.
6. As a developer, I want the aggregation tool to be whitelist-driven (metric/group-by enums, validated condition), so that the LLM can never emit arbitrary or unsafe SQL.
7. As a developer, I want aggregates computed SQL-side over the full matching set, so that no aggregate is ever silently truncated by the row LIMIT.


## Implementation Decisions

### `contracts_aggregate` tool (modules: `apps/search/langchain_agent.py` + `apps/search/where_sql.py` + `apps/search/service.py`)

**Signature (structured, no free SQL):**

```python
contracts_aggregate(
    metric: str,        # "count" | "sum_amount" | "avg_amount"
    group_by: str = "", # "" | "department" | "counterparty_name" | "contract_type" | "status_label" | "year"
    condition: str = "" # natural-language filter, via condition_to_sql
) -> str
```

**Whitelists (module-level constants):**

- `_AGG_METRICS`: `count` → `COUNT(*)`, `sum_amount` → `SUM(CAST(json_extract(tags,'$.amount') AS REAL))`, `avg_amount` → `AVG(CAST(json_extract(tags,'$.amount') AS REAL))`.
- `_AGG_GROUPS`: maps each `group_by` label to its `json_extract` expression; `year` maps to `strftime('%Y', date(json_extract(tags,'$.contract_end_date')))`. Empty `group_by` ⇒ a single overall figure, no `GROUP BY`.

**Query construction (new helper in `where_sql.py`, e.g. `aggregate_sql(metric, group_by, condition) -> Optional[str]`):**

- Resolve `metric`/`group_by` through the whitelists; unknown value ⇒ return `None` (caller surfaces an "unsupported metric/group" observation, never raises).
- Build `SELECT {group_expr} AS k, {agg_expr} AS v FROM sections` (+ `WHERE {cond}` when the condition translates) `GROUP BY k ORDER BY v DESC`. The WHERE fragment reuses `condition_to_sql(condition)`'s translation rather than re-implementing it.
- Pass the assembled SQL through the existing `_validate_sql` (read-only, single statement, `sections` only, `_FORBIDDEN` blocklist). The auto-`LIMIT` then bounds *groups returned*, not rows scanned — the correct bounding.

**Execution (new method on the unified service, e.g. `search_service.aggregate(...) -> str`):**

- Run the validated SQL via the same `sections` connection used by `_run_sections_sql`.
- Render a compact table: header line (`"Total contract amount by department (N groups):"`), aligned `group | value` columns, formatted amounts (`HK$45.2M` style, reusing the existing amount-label convention), and a `"+N more groups"` overflow marker past ~50 groups. Empty result ⇒ a clear "no matching contracts" string.

**Tool wiring:**

- `build_langchain_tools(contract_tool, where_tool, aggregate_tool=None)` gains an optional `aggregate_tool` param, mirroring `where_tool`. When provided, append the `contracts_aggregate` `@tool` with a docstring steering the LLM: *use for counts, totals, averages, per-group breakdowns, "how many", "which X has the most/highest", "compare … across/by X".*
- Add `TOOL_CONTRACTS_AGGREGATE = "contracts_aggregate"` to `apps/search/intents.py` alongside `TOOL_CONTRACT_SEARCH`/`TOOL_CONTRACTS_WHERE`.

### `rank_by` param on `contract_search` (modules: `apps/search/service.py` + `langchain_agent.py`)

- `search(..., rank_by=...)` already accepts a rank hint (`RANK_RELEVANCE`/risk). Surface it to the LLM as a whitelisted param `rank_by: str = "relevance" | "risk" | "amount"` on the `contract_search` `@tool`.
- `rank_by="amount"` sorts the candidate set by `CAST(json_extract(tags,'$.amount') AS REAL)` descending; `"risk"` uses the existing risk-score ordering (already unconditional); `"relevance"` is the current default.
- Update the `contract_search` docstring to document `rank_by` so the LLM knows it exists for "top N by amount/risk" phrasings.

## Out of Scope (explicitly deferred)

- `contract_detail` tool (single-contract drill-down).
- `contracts_compare` tool — the LLM emulates it via two `contracts_aggregate` calls with different conditions.
- Additional metrics (`min`/`max`/median) and additional group-bys beyond the whitelist.
- Time-series/chart rendering in the UI — the observation stays a text table.
- The fast-route comparative gate's keyword list (owned by the prior spec; unchanged here).

## Testing Decisions

Add tests in `tests/test_contracts_where.py` (aggregation, alongside the existing where-SQL tests) and `tests/test_langchain_agent.py` (tool wiring/routing). No live DB/LLM: drive via the existing scripted-LLM and a stub service connection.

- **`aggregate_sql` unit tests** — each whitelisted metric × each group-by produces a validated read-only SELECT with the correct `json_extract`/`GROUP BY`; unknown metric or group-by returns `None`; a `condition` is translated and injected into the WHERE clause.
- **Safety tests** — the assembled aggregate SQL passes `_validate_sql` (starts with SELECT, single statement, sections only, no `_FORBIDDEN` keywords).
- **Truncation-correctness test** — aggregation runs over the full matching set (assert the SQL is an aggregate, not a row-select followed by Python summation).
- **Tool routing tests** — extend `test_langchain_agent.py`: a scripted LLM calling `contracts_aggregate` for "how many contracts per department" invokes the aggregate tool (and not `contract_search`); "top 5 by amount" sets `rank_by="amount"`.
- **Observation format test** — `aggregate()` renders the bounded table with header, aligned columns, and the "+N more" overflow marker.
- Full suite (`pytest`) must stay green.

## Open Questions

None — the five decisions above were explicitly confirmed in the grilling session.
