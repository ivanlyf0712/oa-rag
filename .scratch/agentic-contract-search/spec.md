## Problem Statement

oa-rag already has a working contract search base and a separate risk planner, but its search experience is still a monolith and lacks the agentic decision layer that makes CorpChat-RAG easier to use. Users need a contract-domain search experience that can decide when to search, when to clarify, how to retune retrieval for contract language, and how to degrade safely when LLM support is unavailable.

The current setup also creates structural friction: the search entrypoint collides with the package name we want for a modular search stack, the stale agent tests still point at CorpChat-RAG paths, and the existing contract risk planner is not yet integrated into a broader agentic routing strategy.

## Solution

Introduce a contract-domain agentic search layer in oa-rag by modularizing the search stack, porting the agentic decision components that are needed for a first shippable slice, and adapting them to oa-rag's contract glossary and structural graph rules.

The new experience should:
- keep the contract base retrieval behavior intact,
- route queries through a contract-aware search gate,
- classify and normalize intent in the contract domain,
- support a limited cross-table agent strategy that combines contract search and risk search,
- preserve deterministic fallbacks when LLM support is absent,
- and remain testable at the highest public seams.

## User Stories

1. As a contract analyst, I want the search stack to be modularized, so that the agentic layer can evolve without a monolithic search file.
2. As a contract analyst, I want a dedicated search package and a separate CLI entrypoint, so that imports are unambiguous and stable.
3. As a contract analyst, I want the agent to decide whether a query should be searched or answered directly, so that simple greetings or small talk do not trigger retrieval.
4. As a contract analyst, I want search routing to understand contract language, so that prompts about agreements, clauses, renewals, liability, and counterparties are handled naturally.
5. As a contract analyst, I want the agent to classify common user intents, so that search, clarification, and static informational responses are routed consistently.
6. As a contract analyst, I want contract filters to map to contract fields, so that intent-driven narrowing uses contract_type, department, counterparty_name, and date range instead of chat labels.
7. As a contract analyst, I want the agent to understand when a query needs deterministic search and when it needs clarification, so that ambiguous questions do not produce misleading results.
8. As a contract analyst, I want a manual cross-table routing strategy that can use contract search and risk search together, so that I can move from general contract discovery into risk-focused review without waiting for a full entity index.
9. As a contract analyst, I want structural graph expansion to remain append-only, so that base retrieval results are never displaced by graph traversal.
10. As a contract analyst, I want graph expansion to stay tied to real structural relationships, so that the search results are grounded in contract metadata rather than inferred similarity edges.
11. As a contract analyst, I want the agent to degrade safely when LLM support is unavailable, so that search remains usable even during model outages.
12. As a contract analyst, I want greeting and system-info interactions to work without search, so that the agent remains responsive for non-retrieval interactions.
13. As a contract analyst, I want ambiguous or unsupported queries to fail softly, so that I receive either a deterministic search fallback or a clarification prompt instead of an error.
14. As a contract analyst, I want contract-domain keyword routing to be tuned to legal and procurement language, so that the agent does not rely on WeCom-style message cues.
15. As a contract analyst, I want the router prompt to speak the contract domain, so that LLM rewrite decisions align with contracts instead of customer chat.
16. As a contract analyst, I want ReAct-style tool routing to reference the available oa-rag tools, so that cross-table reasoning uses the actual contract and risk capabilities.
17. As a contract analyst, I want the existing search regression guarantees to remain green, so that the new agent layer does not regress the base search quality.
18. As a contract analyst, I want a contract-domain agent regression gate, so that the new behavior is protected at the public seam.
19. As a maintainer, I want tests to exercise the highest public seam, so that implementation details can change without breaking the suite.
20. As a maintainer, I want the stale CorpChat-specific agent tests replaced, so that oa-rag only tests its own contract-domain behavior.
21. As a maintainer, I want a staged rollout plan, so that the feature can ship in increments instead of as a single risky change.
22. As a maintainer, I want the Streamlit Ask (Agent) UI view deferred, so that backend behavior can be validated before the user interface changes.
23. As a maintainer, I want the new spec to respect oa-rag terminology, so that issue names and implementation notes use the project glossary consistently.
24. As a maintainer, I want structural graph behavior to remain consistent with ADR-0001, so that no vector-inferred edges are reintroduced.

## Implementation Decisions

- The search CLI entrypoint will be renamed so the modular search package can own the  namespace cleanly.
- The search stack will be split into a package rather than remaining a single monolith.
- The first port will include the agentic components needed for a usable contract-domain search flow, but the rollout will remain incremental rather than all-at-once.
- The existing contract risk planner will participate in the agentic plan as a paired tool-path rather than being replaced.
- The cross-table strategy will be manual and tool-based first, using contract search and risk search as the available sources of truth.
- The intent model will keep five intents and map filter behavior into contract-domain fields such as contract_type, department, counterparty_name, and date range.
- Router prompts, keyword rules, and tool-routing examples will be rewritten for contract language.
- Structural graph expansion remains governed by ADR-0001: append-only, deterministic, and limited to traversal-eligible structural edges.
- LLM support must be optional. When unavailable, deterministic rules and safe defaults must keep the system usable.
- The agent layer will be validated at public seams rather than by testing private helpers.
- The stale CorpChat-specific agent test module will be replaced by contract-domain tests that import oa-rag modules only.
- The Streamlit Ask (Agent) view is not part of this spec.

## Testing Decisions

- Tests must assert external behavior, not implementation details.
- The highest public seam should be used wherever possible.
- The base search regression suite remains a required regression gate.
- A new contract-domain agent regression suite will replace the stale CorpChat-specific agent tests.
- Prior art already exists in oa-rag for public-seam tests around search behavior, UI behavior, and risk planning.
- Search behavior should continue to be validated through the public search entrypoints and not by inspecting internal helper functions.
- Agent behavior should be validated through the public agent entrypoint and its observable outputs: intent, response, and returned results.
- Graceful degradation should be tested by simulating LLM unavailability and confirming deterministic fallback behavior.

## Out of Scope

- The Streamlit Ask (Agent) UI view.
- A full true cross-table entity index.
- Reintroducing vector-inferred graph edges.
- Rewriting the underlying contract retrieval base.
- Deep internal unit tests for private routing helpers when a higher seam exists.
- Broader product changes outside contract search and the immediate agentic layer.

## Further Notes

This feature should be treated as an incremental backend modernization of oa-rag's contract search experience. The goal is to preserve the current retrieval guarantees while introducing a contract-aware agent layer that can grow safely over time.
