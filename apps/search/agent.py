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
from apps.search.agent_core import AgentCore
from apps.search.intents import (
    TOOL_CONTRACT_SEARCH,
    TOOL_CONTRACTS_WHERE,
    TOOL_NONE,
)
from apps.search.router import SearchRouter
from apps.search.synthesis import EMPTY_OBSERVATION_MESSAGE

logger = logging.getLogger("oa-search.agent")

# Tool callable signatures:
#   contract_tool(query: str, filters: Dict[str, str]) -> str
ContractTool = Callable[[str, Dict[str, str]], str]


class CrossTableAgent(AgentCore):
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
        risk_tool: Optional[Any] = None,  # deprecated/ignored (Candidate 2)
        where_tool: Optional[Callable[[str], str]] = None,
        router: Optional[SearchRouter] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = LITELLM_MODEL,
        profile: Any = None,
        hindsight_bank: Optional[str] = None,
    ):
        super().__init__(contract_tool=contract_tool, where_tool=where_tool,
                         profile=profile, hindsight_bank=hindsight_bank)
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self.router = router or SearchRouter(api_base=self.api_base,
                                             api_key=self.api_key, model=self.model)

    # ── routing engine (Candidate 1: manual SearchRouter) ────────
    def decide(self, query: str, history, add_step) -> Dict[str, Any]:
        """Manual-ReAct routing: a single SearchRouter.decide call."""
        return self.router.decide(query)

    def _normalize_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Routing rule (ticket 05): 'list all'-style queries whose filter
        content is rule-expressible (or empty) go to the structured path
        (contracts_where), never to vector search. Other enumeration queries
        still hit contract_search, whose service applies its own structured-path
        routing internally."""
        if not self._has_where_tool:
            return decision
        from apps.search.service import _is_enumeration_query
        from apps.search.where_sql import _condition_to_sql, enumeration_remainder
        query = decision.get("query") or ""
        if _is_enumeration_query(query):
            remainder = enumeration_remainder(query)
            if not remainder or _condition_to_sql(remainder) is not None:
                decision = dict(decision)
                decision["tool"] = TOOL_CONTRACTS_WHERE
        return decision

    def _summarize(self, query: str, tool: str, observation: str) -> str:
        return self._synthesize(query, tool, observation)

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
