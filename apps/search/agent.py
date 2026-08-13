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
from typing import Any, Callable, Dict, List, Optional

from apps.search._core import LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL
from apps.search.intents import (
    INTENT_CLARIFY,
    TOOL_CONTRACT_SEARCH,
    TOOL_CONTRACTS_WHERE,
    TOOL_NONE,
    TOOL_RISK_SEARCH,
)
from apps.search.router import SearchRouter

logger = logging.getLogger("oa-search.agent")

# Tool callable signatures:
#   contract_tool(query: str, filters: Dict[str, str]) -> str
#   risk_tool(query: str) -> str
ContractTool = Callable[[str, Dict[str, str]], str]
RiskTool = Callable[[str], str]


def _missing_contract_tool(query: str, filters: Dict[str, str]) -> str:
    raise RuntimeError("contract_search tool is not configured")


def _missing_risk_tool(query: str) -> str:
    raise RuntimeError("risk_search tool is not configured")


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
    ):
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self.contract_tool: ContractTool = contract_tool or _missing_contract_tool
        self._has_risk_tool = risk_tool is not None
        self.risk_tool: RiskTool = risk_tool or _missing_risk_tool
        self._has_where_tool = where_tool is not None
        self.where_tool = where_tool
        self.router = router or SearchRouter(api_base=self.api_base,
                                             api_key=self.api_key, model=self.model)
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

        # ── Step 1: routing decision (degrades safely without LLM) ──
        _stage("🧠", "routing...")
        decision = self.router.decide(query)
        intent = decision.get("intent", "general")
        tool = decision.get("tool", TOOL_CONTRACT_SEARCH)
        retrieval_query = decision.get("query") or query
        filters = decision.get("filters") or {}
        used_fallback = not bool(decision.get("raw"))
        if tool == TOOL_RISK_SEARCH and not self._has_risk_tool:
            # Risk search is not wired on this agent; degrade to the safe
            # default (contract search) instead of raising.
            tool = TOOL_CONTRACT_SEARCH
            used_fallback = True
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
        self._add_step("🧠", "Routing", f"intent={intent}, tool={tool}")

        # ── Step 2: clarify / no-search → answer directly ──
        if not decision.get("search", True) or tool == TOOL_NONE:
            clarification = decision.get("clarification_question") or                 "Could you tell me which contract, counterparty, or risk area you mean?"
            self._add_step("💬", "Clarify", clarification[:60])
            return {
                "output": clarification,
                "intent": intent,
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
            elif tool == TOOL_RISK_SEARCH:
                _stage("🔍", f"risk_search... query: {retrieval_query}")
                observation = self.risk_tool(retrieval_query)
                self._add_step("🔍", "risk_search", f"Query: '{retrieval_query}'")
                tool_calls.append({"tool": TOOL_RISK_SEARCH,
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
        if not observation or not observation.strip():
            return "No matching contracts were found."
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
        prompt = (
            f"You are a contract search assistant. Using ONLY the {source} results below, "
            f"answer the user's question concisely in the same language as the question. "
            f"If the results are empty, say so honestly.\n\n"
            f"Question: {query}\n\nResults:\n{observation}\n\nAnswer:"
        )
        client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key,
                               model=self.model)
        content = client.chat([{"role": "user", "content": prompt}],
                              temperature=0.1, max_tokens=512, timeout=30).strip()
        return content or observation
