"""Shared agent orchestration core (Candidate 1).

``CrossTableAgent`` and ``LangChainAgent`` previously each implemented the same
routing→dispatch→synthesize pipeline twice. This module is the single deep
module that owns that skeleton:

* empty-query guard and process-timeline bookkeeping,
* on-demand Hindsight cross-session memory recall (决策 16/17),
* tool dispatch to the unified ``contract_search`` tool or ``contracts_where``,
* clarification short-circuit, and the answer-synthesis step.

The two agents differ only in **how a routing decision is produced** and in
**how an answer is summarized**, so those are the polymorphic seams:

* :meth:`AgentCore.decide` — return a routing decision dict for the query.
  A decision carries ``_answer`` when the engine already produced a final
  natural-language answer (e.g. the ReAct loop or a greeting fast-path) so the
  core can return it directly.
* :meth:`AgentCore._summarize` — turn a tool observation into an answer.
* :meth:`AgentCore._normalize_decision` — engine-specific tool/intent cleanup.

The result dict exposes the union of both historical shapes (``clarify`` and
``observation`` are always present); the Streamlit UI reads every key with
``.get()``, so both agents share one shape without breaking the app.
"""
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from apps.search.hindsight_client import needs_recall, recall as hs_recall
from apps.search.intents import (
    DEFAULT_CLARIFICATION,
    INTENT_CLARIFY,
    TOOL_CONTRACT_SEARCH,
    TOOL_CONTRACTS_WHERE,
    TOOL_NONE,
)
from apps.search.synthesis import EMPTY_OBSERVATION_MESSAGE

logger = logging.getLogger("oa-search.agent")


def _missing_contract_tool(query: str, filters: Dict[str, str]) -> str:
    raise RuntimeError("contract_search tool is not configured")


class AgentCore:
    """Orchestration skeleton shared by the contract-search agents."""

    # Message used when the user submits an empty query (single source).
    EMPTY_QUERY_MESSAGE = "Please enter a contract search query."

    def __init__(
        self,
        contract_tool: Optional[Callable[[str, Dict[str, str]], str]] = None,
        where_tool: Optional[Callable[[str], str]] = None,
        profile: Any = None,
        hindsight_bank: Optional[str] = None,
        clarification: str = DEFAULT_CLARIFICATION,
    ):
        self.contract_tool = contract_tool or _missing_contract_tool
        self.where_tool = where_tool
        self._has_where_tool = where_tool is not None
        self.profile = profile
        # Hindsight 记忆银行 ID; None/空 → 不做跨会话记忆 recall
        self.hindsight_bank = hindsight_bank or os.getenv("HINDSIGHT_BANK_ID") or None
        self._clarification = clarification
        self._steps: List[Dict[str, Any]] = []

    # ── process timeline ─────────────────────────────────────────
    def _add_step(self, icon: str, label: str, detail: str = "") -> None:
        self._steps.append({"icon": icon, "label": label, "detail": detail})

    # ── result shape (union of both historical seams) ────────────
    def _result(self, *, output, intent, tool, tool_calls, success, fallback,
                clarify, observation) -> Dict[str, Any]:
        return {
            "output": output, "intent": intent, "tool": tool,
            "tool_calls": tool_calls, "steps": self._steps, "success": success,
            "fallback": fallback, "clarify": clarify, "observation": observation,
        }

    # ── polymorphic seams (subclasses override) ──────────────────
    def decide(self, query: str, history, add_step) -> Dict[str, Any]:
        """Produce a routing decision. Must be overridden."""
        raise NotImplementedError

    def _summarize(self, query: str, tool: str, observation: str) -> str:
        """Turn a tool observation into an answer. Overridden per agent."""
        return observation or EMPTY_OBSERVATION_MESSAGE

    def _normalize_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Engine-specific tool/intent cleanup; identity by default."""
        return decision

    def _prepare_filters(self, retrieval_query: str, query: str,
                         filters: Dict[str, Any]) -> Dict[str, Any]:
        """Hook to enrich filters before dispatch; identity by default."""
        return filters

    # ── shared: Hindsight on-demand recall ───────────────────────
    def _recall_memories(self, query: str) -> Tuple[str, Optional[str]]:
        """Return (agent_input, detail-or-None). Injects cross-session memories
        into the routing input only when an explicit recall trigger word is
        present; otherwise returns the query unchanged and no detail.
        """
        agent_input = query
        detail = None
        if not (self.hindsight_bank and needs_recall(query)):
            return agent_input, detail
        t0 = time.perf_counter()
        memories: List[Dict[str, Any]] = []
        try:
            memories = hs_recall(query, bank=self.hindsight_bank, max_results=5)
        except Exception as e:
            logger.debug("Hindsight recall failed: %s", e)
        ms = int((time.perf_counter() - t0) * 1000)
        detail = "recall on '%s' (%d memories, %dms)" % (query[:40], len(memories), ms)
        if memories:
            lines = []
            for m in memories:
                content = str(m.get("content") or "").strip()
                if content:
                    lines.append("- " + content)
            if lines:
                agent_input = (
                    query
                    + "\n\n[Relevant cross-session memories (Hindsight bank %s)]:\n%s\n"
                      "[Use these memories if relevant; ignore otherwise.]"
                    % (self.hindsight_bank, "\n".join(lines))
                )
        return agent_input, detail

    # ── shared: clarify short-circuit ────────────────────────────
    def _clarify_result(self, clarification: str, intent: str, fallback: bool) -> Dict[str, Any]:
        self._add_step("💬", "Clarify", clarification[:60])
        return self._result(
            output=clarification, intent=intent, tool=TOOL_NONE, tool_calls=[],
            success=True, fallback=fallback, clarify=True, observation="",
        )

    # ── shared: tool dispatch + synthesis ────────────────────────
    def _run_tool(self, query: str, retrieval_query: str, tool: str,
                  filters: Dict[str, Any], intent: str, fallback: bool,
                  stage) -> Dict[str, Any]:
        tool_calls: List[Dict[str, Any]] = []
        observation = ""
        try:
            if tool == TOOL_CONTRACTS_WHERE and self._has_where_tool:
                stage("🔍", "contracts_where... condition: %s" % retrieval_query)
                observation = self.where_tool(retrieval_query)
                self._add_step("🔍", "contracts_where", "Condition: '%s'" % retrieval_query)
                tool_calls.append({"tool": TOOL_CONTRACTS_WHERE,
                                   "tool_input": retrieval_query, "filters": {},
                                   "observation": observation[:200]})
            else:
                stage("🔍", "contract_search... query: %s" % retrieval_query)
                filters = self._prepare_filters(retrieval_query, query, dict(filters))
                observation = self.contract_tool(retrieval_query, filters)
                self._add_step("🔍", "contract_search", "Query: '%s'" % retrieval_query)
                tool_calls.append({"tool": TOOL_CONTRACT_SEARCH,
                                   "tool_input": retrieval_query, "filters": filters,
                                   "observation": observation[:200]})
        except Exception as e:
            logger.warning("Agent tool failed: %s", e)
            self._add_step("⚠️", "Tool error", str(e)[:100])
            return self._result(
                output="I could not complete the search: %s" % e, intent=intent,
                tool=tool, tool_calls=tool_calls, success=False, fallback=True,
                clarify=False, observation=observation,
            )

        stage("✨", "generating answer...")
        self._add_step("✨", "Answer generation", "Combining results")
        output = self._summarize(query, tool, observation)
        return self._result(
            output=output, intent=intent, tool=tool, tool_calls=tool_calls,
            success=True, fallback=fallback, clarify=False, observation=observation,
        )

    # ── main entry ───────────────────────────────────────────────
    def process(self, user_input: str, on_stage: Optional[Callable] = None,
                on_tool: Optional[Callable] = None,
                history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Route user_input to a tool and synthesize an answer."""

        def stage(label: str, detail: str = ""):
            if on_stage:
                try:
                    on_stage(label, detail)
                except Exception:
                    pass

        self._steps = []
        query = (user_input or "").strip()
        if not query:
            return self._result(
                output=self.EMPTY_QUERY_MESSAGE, intent=INTENT_CLARIFY,
                tool=TOOL_NONE, tool_calls=[], success=False, fallback=False,
                clarify=True, observation="",
            )

        # ── Step 1: classify the query ──
        stage("🧭", "routing...")
        self._add_step("🧭", "Routing", "Query: '%s'" % query[:60])

        # On-demand cross-session memory recall (决策 16/17).
        agent_input, mem_detail = self._recall_memories(query)
        if mem_detail is not None:
            self._add_step("🧠", "Hindsight memory", mem_detail)

        # ── Step 2: produce a routing decision via the engine ──
        decision = self.decide(agent_input, history, self._add_step)
        if decision.get("_answer") is not None:
            # The engine already produced a final answer (e.g. the ReAct loop
            # or a greeting fast-path); return it directly.
            return decision["_answer"]
        decision = self._normalize_decision(decision)
        intent = decision.get("intent", INTENT_CLARIFY)
        tool = decision.get("tool", TOOL_CONTRACT_SEARCH)
        retrieval_query = decision.get("query") or agent_input
        filters = decision.get("filters") or {}
        used_fallback = not bool(decision.get("raw"))
        self._add_step("🧭", "Routing", "intent=%s, tool=%s" % (intent, tool))

        # ── Step 3: clarify / no-search → answer directly ──
        if not decision.get("search", True) or tool == TOOL_NONE:
            clarification = decision.get("clarification_question") or self._clarification
            return self._clarify_result(clarification, intent, used_fallback)

        # ── Step 4: run the chosen tool + synthesize ──
        return self._run_tool(query, retrieval_query, tool, filters, intent,
                              used_fallback, stage)
