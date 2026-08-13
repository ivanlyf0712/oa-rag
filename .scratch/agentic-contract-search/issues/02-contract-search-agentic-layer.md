Status: ready-for-agent
Type: task

# 02 — Contract-domain agentic search layer

**What to build:** the contract-aware agentic search behavior that decides when to search, how to rewrite or clarify a query, and how to route between contract search and risk search.

**Blocked by:** 01 — Search package boundary and CLI split.

## Acceptance criteria
- The search gate understands contract language.
- The intent model keeps the agreed five intents.
- Filter behavior maps to contract-domain fields.
- Router and decision logic degrade safely when LLM support is unavailable.
- The manual cross-table path can route between contract search and risk search.
- Domain-specific prompts, keywords, and tool-routing language are retuned for contract search.
- The agentic layer remains compatible with the existing contract search base and structural graph rules.
