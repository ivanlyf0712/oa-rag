Status: ready-for-agent
Type: task

# 03 — One unified contract-search tool (expand phase)

**What to build:** a single tool callable that accepts the planner's flat filter list (ticket 01) and returns the deterministic, `risk_level`-enriched results from ticket 02. It is wired into the CLI tool builders and shared by both agents (manual ReAct and LangChain), so both route through the same planner and the same filter path. This is the *expand* phase of retiring the risk fork: the legacy risk tool is left in place so the running app keeps working while consumers migrate. A query like "find completed contracts with high risk" now filters deterministically on `status=completed` plus the high-risk threshold and shows the `risk_level` column, with no semantic drift.

**Blocked by:** 02 — Deterministic tag-filter search.

## Acceptance criteria
- A single contract-search tool exists that takes a filter list and returns filtered + risk-scored results.
- Both the manual ReAct agent and the LangChain agent obtain results through this one tool and the shared planner — routing cannot diverge between agents.
- "Find completed contracts with high risk" produces a deterministic `status=completed` + high-risk-threshold filter, sorted by score, each row showing `risk_level`.
- "High risk" filtering uses the same scoring function and threshold as the displayed `risk_level` label, so filter and display cannot disagree.
- The legacy risk tool still runs (expand phase) so the existing app does not break.
- Tool behaviour is verified with a stubbed planner and stubbed `sections` table — no live LLM or index.
