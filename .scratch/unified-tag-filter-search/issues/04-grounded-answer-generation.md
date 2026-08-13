Status: ready-for-agent
Type: task

# 04 — Grounded answer generation (verdict from tags, flag contradictions)

**What to build:** the answer-synthesis contract changes so the generated answer is grounded and trustworthy. The synthesis prompt is given (i) the deterministic verdict computed from tags — risk level, status, field values — verbatim, (ii) the supporting facts from the chosen contract's attachment summary, and (iii) an explicit instruction to state the verdict plainly, report the attachment facts, and flag any contradiction between the attachment text and the tags without judging or overriding them. The LLM never derives the verdict itself; it only narrates what the deterministic tags already decided. Per-contract attachment analysis (selection/extraction/summarization) plugs into this stage unchanged.

**Blocked by:** 03 — One unified contract-search tool.

## Acceptance criteria
- The synthesis prompt carries the tag-derived verdict verbatim and an explicit contradiction-flagging instruction.
- The generated answer states the verdict (e.g. risk level, status) plainly and reproducibly, matching the deterministic tags.
- When the contract document contradicts a tag value, the answer flags the discrepancy rather than silently smoothing it over.
- Supporting facts are drawn from the chosen contract's attachment summary.
- Verified with a stubbed LLM: the prompt contains the verdict + flag instruction and the output states the verdict — no live LLM required.
