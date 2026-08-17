# Fast-Route Comparative Gate

**Spec status:** ready-for-agent
**Source:** grilling session 2026-08-17; all decisions below were explicitly confirmed by the user.

## Problem Statement

`LangChainAgent.process` runs four engines in sequence — greeting fast-path, `_fast_route`, the LangGraph ReAct loop, and the deterministic router fallback. `_fast_route` exists for one reason: skip an expensive ReAct LLM round-trip when the routing decision is *certain*.

Two fast-route paths exist today:

- **Bare ref-number queries** (`"CCA20250096"`) → `contract_search`. This path is already pure: `_looks_like_ref_no` uses `_REF_NO_RE.fullmatch`, so any extra words break the match and the query falls through. No leak here.
- **Risk-keyword queries** (`"risk not accepted"`, `"needs legal review"`) → `contract_search` (risk intent preserved). This path is **leaky**: `infer_intent_from_query` returns `INTENT_RISK` on a mere substring match, with no regard for whether the query is a pure retrieval or an analytical/comparative one.

So a query like `"compare risk-not-accepted contracts across departments"` fast-routes to a single deterministic `contract_search` call — exactly the multi-step / synthesising case the ReAct engine is meant to handle. The fast-route pre-emption, designed to save an LLM call only when confident, is silently swallowing comparative risk queries and short-circuiting the engine the user actually wants to exercise.

## Solution

Tighten the risk branch of `_fast_route` with a **comparative/aggregate keyword gate**: when a risk phrase co-occurs with an analytical signal, `_fast_route` returns `None` and the query proceeds to the ReAct loop. Pure risk retrievals keep fast-routing (the latency saving is preserved for the common case). The bare-ref branch is untouched.

Decisions locked in the grilling session:

1. **Keep the ReAct engine** — it is the intended production engine (`langgraph==1.2.11`, `langchain-core==1.5.3` are pinned in `requirements.txt`; the docker image builds with them). The bare-metal venv lacks the deps, so the loop is unreachable *there*, but it is planned-to-run, not dead code. No engine deletion.
2. **Tighten, don't drop, `_fast_route`** — the pre-emption is worth keeping for pure ref/risk retrievals; only the analytical leak needs closing.
3. **Comparative/aggregate keyword gate** — detection is a deterministic keyword list, matching the existing "skip an LLM call only when certain" spirit. The bare-ref branch is already pure via `fullmatch` and needs no change.
4. **Comparative always wins** — any comparative/aggregate keyword co-occurring with a risk phrase returns `None`. No per-keyword precedence tiers, no order-dependence. The moment an analytical signal appears we are no longer *certain* of a single tool call, so we yield.

## User Stories

1. As a contracts reviewer, I want comparative risk questions ("compare risk-not-accepted contracts by department") answered by the ReAct engine, so that I get multi-step reasoning rather than a single flat retrieval.
2. As a contracts reviewer, I want simple risk lookups ("show contracts where risk was not accepted") to stay fast, so that I'm not paying an LLM round-trip for a deterministic filter.
3. As a contracts reviewer, I want bare ref-number lookups ("CCA20250096") to keep fast-routing, so that exact lookups stay instant regardless of this change.
4. As a developer, I want the comparative gate to be a single deterministic keyword list in one place, so that the fast-route boundary is easy to audit and extend.
5. As a developer, I want the golden routing matrix extended with comparative-yield cases, so that a future regression that re-leaks analytical risk queries is caught immediately.


## Implementation Decisions

### Comparative gate (module: `apps/search/langchain_agent.py`)

- Add a module-level constant `_ANALYTICAL_SIGNALS`: a tuple of lowercase comparative/aggregate needles — `compare`, `vs`, `versus`, `trend`, `over time`, `why`, `how many`, `breakdown`, `summarize`, `average`, `total`, `group by`, plus the superlatives `most` and `highest` (which signal "which X has the most/highest …").
- Add a helper `_looks_analytical(query: str) -> bool` returning `True` when any needle appears in the lower-cased query.
- In `_fast_route`, **only the risk branch** is gated:

  ```python
  intent = infer_intent_from_query(query)
  if intent == INTENT_RISK:
      if _looks_analytical(query):
          return None            # comparative risk -> yield to ReAct
      tool = TOOL_CONTRACT_SEARCH
  elif _looks_like_ref_no(query):
      tool = TOOL_CONTRACT_SEARCH
      intent = INTENT_GENERAL
  else:
      return None
  ```

- The bare-ref branch is **not** gated: `_REF_NO_RE.fullmatch` already guarantees the query is nothing but a ref number, so no analytical signal can co-occur.
- Precedence is unconditional: a comparative signal plus a risk phrase returns `None`. No "soft" analytical whitelist, no positional/order logic.
- No new imports, no new dependencies, no change to `infer_intent_from_query` (it is shared with the deterministic fallback router and must keep its current substring semantics). The gate lives entirely inside `_fast_route`.

### Out of scope

- The ReAct loop's internal reasoning, tool wrapping, recursion limit, and the router fallback — unchanged.
- The greeting fast-path (`_quick_respond`) — unchanged.
- `CrossTableAgent` — it has no fast-route tier; its behaviour is unaffected.

## Testing Decisions

Extend the existing seams in `tests/test_langchain_agent.py` (the `ScriptedLLM` + `ROUTING_SCRIPT` fixture and the `GOLDEN_CASES` matrix). No new fixtures are needed.

- **Golden matrix stays green** — the existing cases `"show contracts where risk was not accepted"` and `"risk not accepted contracts"` (both `INTENT_RISK`) are pure retrievals and must keep fast-routing.
- **Dedicated `_fast_route` tests** (cleaner than the golden matrix for yield behaviour, since a yielded query continues to the scripted path):
  - `test_fast_route_yields_comparative_risk` — parametrize over `"compare risk-not-accepted contracts across departments"`, `"risk not accepted vs accepted counts"`, `"why was risk not accepted"`, `"how many risk not accepted contracts by month"`; assert `agent._fast_route(query) is None`.
  - `test_fast_route_keeps_pure_risk` — assert `agent._fast_route("risk not accepted contracts")` returns a decision with `tool == TOOL_CONTRACT_SEARCH` and `intent == INTENT_RISK` (guards against over-yielding).
  - `test_fast_route_keeps_bare_ref` — assert `agent._fast_route("CCA20250096")` still returns a `contract_search` / `INTENT_GENERAL` decision (ref branch unaffected).
- Full suite (`pytest`) must stay green; no snapshot/output changes are expected since fast-route output is internal.

## Open Questions

None — the four decisions above were explicitly confirmed in the grilling session.
