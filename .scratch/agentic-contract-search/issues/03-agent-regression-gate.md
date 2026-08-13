Status: ready-for-agent
Type: task

# 03 — Contract-domain agent regression gate

**What to build:** regression coverage for the public agent seam that proves the new contract-domain agent behavior remains stable, including graceful degradation when LLM support is unavailable.

**Blocked by:** 01 — Search package boundary and CLI split; 02 — Contract-domain agentic search layer.

## Acceptance criteria
- Tests cover the public agent seam.
- Tests assert only external behavior.
- Existing base search regressions remain green.
- LLM-down behavior is deterministic and covered.
- The stale CorpChat-specific agent tests are replaced or superseded.
- The tests reflect the contract glossary and do not encode brittle implementation paths.
