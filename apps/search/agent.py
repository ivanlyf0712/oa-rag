"""Contract search — CrossTableAgent (manual ReAct router).

A manual-ReAct router over the unified contract search pipeline. The agent
classifies the query, maps filters (contract type, counterparty, status,
expired, contract id, dates) onto the contract corpus, runs the contract
search tool, and synthesizes an answer. No LangChain dependency; the loop is
explicit and degrades safely when the LLM is unavailable.

Flow:
  User query → SearchRouter.decide() → contract_search → synthesize answer.

Tools are injected as callables so the agent is fully testable without a real
index, database, or LLM.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from apps.search._core import LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL
from apps.search.hindsight_client import needs_recall, recall as hs_recall
from apps.search.intents import (
    INTENT_CLARIFY,
    TOOL_CONTRACT_SEARCH,
    TOOL_CONTRACTS_WHERE,
    TOOL_NONE,
)
from apps.search.router import SearchRouter
from apps.search.synthesis import EMPTY_OBSERVATION_MESSAGE

logger = logging.getLogger("oa-search.agent")

# Tool callable signatures:
#   contract_tool(query: str, filters: Dict[str, str]) -> str
#   risk_tool(query: str) -> str
ContractTool = Callable[[str, Dict[str, str]], str]
RiskTool = Callable[[str], str]


def _missing_contract_tool(query: str, filters: Dict[str, str]) -> str:
    raise RuntimeError("contract_search tool is not configured")


class CrossTableAgent:
    """Manual-ReAct agent over the unified contract search pipeline.

    Returns a dict with:
      - output: str — final natural-language answer
      - intent: str — classified intent
      - tool: str — tool that was used (or 'none')
      - tool_calls: List[Dict] — executed tool calls (tool, input, observation)
      - steps: List[Dict] — process timeline (icon, label, detail)
      - success: bool
      - fallback: bool — True when the LLM path was unavailable
    """

    def __init__(
        self,
        contract_tool: Optional[ContractTool] = None,
        risk_tool: Optional[RiskTool] = None,
        where_tool: Optional[Callable[[str], str]] = None,
        router: Optional[SearchRouter] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = LITELLM_MODEL,
        profile: Any = None,
        hindsight_bank: Optional[str] = None,
    ):
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self.contract_tool: ContractTool = contract_tool or _missing_contract_tool
        # Candidate 2: risk queries route to the unified contract_search tool;
        # risk scoring/ranking happens inside the search service. The risk_tool
        # parameter is accepted for backward compatibility but ignored.
        self._has_where_tool = where_tool is not None
        self.where_tool = where_tool
        self.router = router or SearchRouter(api_base=self.api_base,
                                             api_key=self.api_key, model=self.model)
        self.profile = profile
        # Hindsight 记忆银行 ID; None/空 → 不做跨会话记忆 recall
        self.hindsight_bank = hindsight_bank or os.getenv("HINDSIGHT_BANK_ID") or None
        self._steps: List[Dict[str, Any]] = []

    # ── process timeline ─────────────────────────────────────────
    def _add_step(self, icon: str, label: str, detail: str = "") -> None:
        self._steps.append({"icon": icon, "label": label, "detail": detail})

    # ── main entry ───────────────────────────────────────────────
    def process(self, user_input: str, on_stage: Optional[Callable] = None) -> Dict[str, Any]:
        """Route a contract-domain query to the right tool and synthesize an answer."""

        def _stage(label: str, detail: str = ""):
            if on_stage:
                try:
                    on_stage(label, detail)
                except Exception:
                    pass

        self._steps = []
        query = (user_input or "").strip()
        if not query:
            return {
                "output": "Please enter a contract search query.",
                "intent": INTENT_CLARIFY,
                "tool": TOOL_NONE,
                "tool_calls": [],
                "steps": self._steps,
                "success": False,
                "fallback": False,
            }

        # ── Step 1: classify the query ──
        _stage("🧭", "routing...")
        self._add_step("🧭", "Routing", f"Query: '{query[:60]}'")

        # Hindsight on-demand recall (决策 16/17): 仅命中显式跨会话引用词
        # (上次/之前/还记得/remember/previously...) 才注入历史记忆; 未命中则静默跳过。
        agent_input = query
        if self.hindsight_bank and needs_recall(query):
            _mem_t0 = time.perf_counter()
            memories: List[Dict[str, Any]] = []
            try:
                memories = hs_recall(query, bank=self.hindsight_bank, max_results=5)
            except Exception as e:
                logger.debug("Hindsight recall failed: %s", e)
            _mem_ms = int((time.perf_counter() - _mem_t0) * 1000)
            self._add_step(
                "🧠", "Hindsight memory",
                "recall on '%s' (%d memories, %dms)" % (query[:40], len(memories), _mem_ms),
            )
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

        # Original routing logic (preserved from HEAD):
        # - fallback detection via raw LLM output presence
        # - risk tool → contract search degradation when risk tool is not wired
        # - contracts_where routing for enumeration queries (ticket 05)
        decision = self.router.decide(agent_input)
        intent = decision.get("intent", INTENT_CLARIFY)
        tool = decision.get("tool", TOOL_CONTRACT_SEARCH)
        retrieval_query = decision.get("query") or agent_input
        filters = decision.get("filters") or {}
        used_fallback = not bool(decision.get("raw"))
        if self._has_where_tool:
            # Routing rule (ticket 05): "list all"-style queries whose filter
            # content is rule-expressible (or empty) go to the structured
            # path (contracts_where), never to vector search. Other
            # enumeration queries still hit contract_search, whose service
            # applies its own structured-path routing internally.
            from apps.search.service import _is_enumeration_query
            from apps.search.where_sql import _condition_to_sql, enumeration_remainder
            if _is_enumeration_query(query):
                remainder = enumeration_remainder(query)
                if not remainder or _condition_to_sql(remainder) is not None:
                    tool = TOOL_CONTRACTS_WHERE
        self._add_step("🧭", "Routing", f"intent={intent}, tool={tool}")

        # ── Step 2: clarify / no-search → answer directly ──
        if not decision.get("search", True) or tool == TOOL_NONE:
            clarification = decision.get("clarification_question") or "Please provide more details."
            self._add_step("💬", "Clarify", clarification[:60])
            return {
                "output": clarification,
                "intent": INTENT_CLARIFY,
                "tool": TOOL_NONE,
                "tool_calls": [],
                "steps": self._steps,
                "success": True,
                "fallback": used_fallback,
            }

        # ── Step 3: run the chosen tool ──
        tool_calls: List[Dict[str, Any]] = []
        observation = ""
        try:
            if tool == TOOL_CONTRACTS_WHERE and self._has_where_tool:
                _stage("🔍", f"contracts_where... condition: {retrieval_query}")
                observation = self.where_tool(retrieval_query)
                self._add_step("🔍", "contracts_where", f"Condition: '{retrieval_query}'")
                tool_calls.append({"tool": TOOL_CONTRACTS_WHERE,
                                   "tool_input": retrieval_query,
                                   "filters": {},
                                   "observation": observation[:200]})
            else:
                _stage("🔍", f"contract_search... query: {retrieval_query}")
                observation = self.contract_tool(retrieval_query, filters)
                self._add_step("🔍", "contract_search", f"Query: '{retrieval_query}'")
                tool_calls.append({"tool": TOOL_CONTRACT_SEARCH,
                                   "tool_input": retrieval_query,
                                   "filters": filters,
                                   "observation": observation[:200]})
        except Exception as e:
            logger.warning("Agent tool failed: %s", e)
            self._add_step("⚠️", "Tool error", str(e)[:100])
            return {
                "output": f"I could not complete the search: {e}",
                "intent": intent,
                "tool": tool,
                "tool_calls": tool_calls,
                "steps": self._steps,
                "success": False,
                "fallback": True,
            }

        # ── Step 4: synthesize the answer ──
        _stage("✨", "generating answer...")
        self._add_step("✨", "Answer generation", "Combining results")
        output = self._synthesize(query, tool, observation)

        return {
            "output": output,
            "intent": intent,
            "tool": tool,
            "tool_calls": tool_calls,
            "steps": self._steps,
            "success": True,
            "fallback": used_fallback,
        }

    # ── answer synthesis (LLM with deterministic fallback) ───────
    def _synthesize(self, query: str, tool: str, observation: str) -> str:
        # Shared empty-result message (single source in apps.search.synthesis).
        if not observation or not observation.strip():
            return EMPTY_OBSERVATION_MESSAGE
        try:
            return self._llm_summarize(query, tool, observation)
        except Exception as e:
            logger.debug("Answer synthesis fell back: %s", e)
            return observation

    def _llm_summarize(self, query: str, tool: str, observation: str) -> str:
        # Route through the shared LiteLLMClient (single LLM entry point);
        # on failure chat() returns "" and we return the raw observation.
        from apps.search.litellm_client import LiteLLMClient
        source = "contract search"

        # Build system prompt with persona instructions when a profile is set.
        system_prompt = (
            f"You are a contract search assistant. Using ONLY the {source} results below, "
            f"answer the user's question concisely in the same language as the question. "
            f"If the results are empty, say so honestly."
        )
        if self.profile is not None:
            try:
                system_prompt = self.profile.build_system_prompt(system_prompt)
            except Exception:
                pass

        user_prompt = (
            f"Question: {query}\n\nResults:\n{observation}\n\nAnswer:"
        )

        client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key,
                               model=self.model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        content = client.chat(messages, temperature=0.1, max_tokens=512, timeout=30).strip()
        return content or observation
