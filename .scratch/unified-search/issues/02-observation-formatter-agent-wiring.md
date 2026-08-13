# 02 — LLM observation formatter, result store & agent wiring

**Spec:** `docs/specs/unified_search_risk_merge.md`

**What to build:** a pure formatter `rows -> observation text` emitting per-row: ref, counterparty/title, type, status label, start/end dates, amount, risk score/severity, top signals, ~200-char snippet — capped at the first 50 numbered rows with an exact `"(+N more contracts not shown - see results table)"` overflow marker when truncated. This supersedes the 5-line truncation in `_default_synthesize` and the old `format_contract_results`/`_format_risk_results` pair. Add a per-invocation result store (pattern: corpchat `tools.snapshot_meta`): tool adapters stash the structured `ContractRow` list + query meta; Streamlit renders from the store. ReWire the agent: LangGraph ReAct stays, `search_contracts` becomes a thin adapter over the unified service, the separate risk tool is retired, and the regex observation-parser in the app is deleted. `fallback` semantics and the loud LangGraph-unavailable warning are unchanged.

**Blocked by:** 01 — Unified search service.

**Status:** done

- [x] Formatter emits dates, amount, status label, risk score/severity/signals per row
- [x] 50-row budget enforced in the formatter; exact overflow marker appended when exceeded
- [x] `_default_synthesize` no longer truncates by line count; empty observation still yields "No matching contracts were found."
- [x] Result store: stash/snapshot round-trip works; invocations are isolated from each other
- [x] ReAct keeps working with the single `search_contracts` adapter; risk tool removed
- [x] Regex observation-parser deleted from the app; UI consumes structured rows from the store
- [x] Tests: formatter fields/budget/marker/empty-set; result store round-trip; agent tests updated (all green)
