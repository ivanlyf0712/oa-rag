Status: ready-for-agent
Type: task

# Contract agentic layer

Port the contract-aware agentic decision layer needed for a first shippable agentic search slice.

## Acceptance Criteria
- The search gate understands contract language.
- The intent model uses the agreed five intents.
- Filter mapping targets contract fields.
- The router and decision logic degrade safely when LLM support is unavailable.
- The cross-table path can route between contract search and risk search.

## Notes
- Keep the initial scope deliberately smaller than a full corpchat-style port.
- Retune all user-facing domain language to the contract glossary.
