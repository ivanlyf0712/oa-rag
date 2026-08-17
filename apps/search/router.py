"""Contract search — LLM Router (search gate + intent classification).

Decides whether a contract-domain message needs retrieval and, if so, which
intent it is and which structured filters should be applied.

Decision format (JSON):
  {
    "search": true/false,
    "intent": one of VALID_INTENTS,
    "query": "rewritten retrieval query",
    "filters": {
      "contract_type": "2",
      "department": "Legal",
      "counterparty_name": "Acme",
      "status": "completed",
      "expired": true,
      "contract_id": "CCA20250096",
      "date_from": "2024-01-01",
      "date_to": "2024-12-31"
    },
    "clarification_question": "..."   # only when intent == clarify
  }

Behavior:
  - JSON parse failure / LLM error → safe default: search the contract corpus.
  - LLM unavailable (no key / network) → degrade safely to default decision.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from apps.search._core import LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL
from apps.search.intents import (
    CONTRACT_FILTER_FIELDS,
    DEFAULT_CLARIFICATION,
    INTENT_CLARIFY,
    INTENT_TO_TOOL,
    VALID_INTENTS,
    default_decision,
    infer_intent_from_query,
    normalize_intent,
)

logger = logging.getLogger("oa-search.router")


class SearchRouter:
    """LLM-driven gate that decides whether/how to search contract data."""

    _SYSTEM_PROMPT = (
        "You are a contract search assistant for a legal/OA system. "
        "Decide whether the user's message requires searching the contract corpus, "
        "classify its intent, and map any filters onto contract fields. "
        "Respond with ONLY a JSON object: "
        "{\"search\": true/false, \"intent\": \"...\", \"query\": \"...\", "
        "\"filters\": {...}, \"clarification_question\": \"...\"}. "
        "Intent must be one of: " + ", ".join(VALID_INTENTS) + ". "
        "Rules: "
        "1) Greetings, thanks, small talk → search=false, intent=general. "
        "2) Questions about contract content, clauses, breach, termination, amounts, "
        "counterparties → search=true. "
        "3) Risk/compliance/flag questions (risk not accepted, needs legal review, "
        "over threshold, external guarantees) → intent=risk (these route to the "
        "dedicated risk_search tool). "
        "4) Questions about a specific company/party → intent=counterparty and set "
        "filters.counterparty_name. "
        "5) Renewal/expiry questions → intent=renewal. "
        "6) Vague or ambiguous input → intent=clarify and provide clarification_question. "
        "7) If search=true, rewrite query into a concise retrieval query. "
        "8) Allowed filter keys: " + ", ".join(CONTRACT_FILTER_FIELDS) + ". "
        "Examples: status=completed (or done/finished), expired=true, "
        "contract_id=CCA20250096, contract_type=2. "
        "Use ISO date strings (YYYY-MM-DD) for date_from/date_to. Omit unknown filters."
    )

    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None,
                 model: str = LITELLM_MODEL):
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ── public ───────────────────────────────────────────────────
    def decide(self, user_message: str) -> Dict[str, Any]:
        """Return a routing decision dict (see module docstring for shape)."""
        query = (user_message or "").strip()
        if not query:
            d = default_decision("")
            d["search"] = False
            d["intent"] = INTENT_CLARIFY
            d["tool"] = INTENT_TO_TOOL[INTENT_CLARIFY]
            d["clarification_question"] = "Please enter a contract search query."
            return d
        if query in self._cache:
            return self._cache[query]

        decision = default_decision(query)
        inferred_intent = infer_intent_from_query(query)
        if inferred_intent != decision["intent"]:
            decision["intent"] = inferred_intent
            decision["tool"] = INTENT_TO_TOOL.get(inferred_intent, INTENT_TO_TOOL[INTENT_CLARIFY])
            if inferred_intent == INTENT_CLARIFY:
                decision["search"] = False
        try:
            raw = self._call_llm([
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ])
            decision["raw"] = raw or ""
            parsed = self._parse_json(raw or "")
            if parsed is not None:
                decision = self._apply(parsed, query)
        except Exception as e:
            logger.debug("Router decision failed: %s", e)

        self._cache[query] = decision
        return decision

    # ── internals ────────────────────────────────────────────────
    def _apply(self, parsed: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Map a parsed LLM JSON object onto the decision contract safely."""
        decision = default_decision(original_query)
        decision["search"] = bool(parsed.get("search", True))

        intent = normalize_intent(parsed.get("intent"))
        decision["intent"] = intent
        decision["tool"] = INTENT_TO_TOOL.get(intent, INTENT_TO_TOOL[INTENT_CLARIFY])

        q = parsed.get("query")
        decision["query"] = q if isinstance(q, str) and q.strip() else original_query

        filters = parsed.get("filters")
        decision["filters"] = self._validate_filters(filters)

        clarification = str(parsed.get("clarification_question") or "").strip()
        if intent == INTENT_CLARIFY:
            decision["search"] = False
            decision["clarification_question"] = clarification or DEFAULT_CLARIFICATION
        return decision

    @staticmethod
    def _validate_filters(filters: Any) -> Dict[str, str]:
        """Keep only known contract filter fields with non-empty values."""
        if not isinstance(filters, dict):
            return {}
        out: Dict[str, str] = {}
        for key in CONTRACT_FILTER_FIELDS:
            value = filters.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out[key] = text
        return out

    def _call_llm(self, messages, max_tokens: int = 300) -> str:
        # 经由共享 LiteLLMClient (单一 LLM 入口); 超时 → 空串 → 确定性路由兜底。
        from apps.search.litellm_client import LiteLLMClient
        try:
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key,
                                   model=self.model)
            return client.chat(messages, temperature=0.0, max_tokens=max_tokens,
                               timeout=30).strip()
        except Exception as e:
            logger.debug("Router LLM call failed: %s", e)
            return ""

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON parse from noisy LLM output."""
        if not text:
            return None
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        return None
