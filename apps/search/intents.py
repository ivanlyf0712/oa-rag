"""Contract-domain intent model.

The agreed five intents (grilling decision #4). Kept deliberately small and
stable — the router maps a user query onto exactly one of these, and the
CrossTableAgent uses the intent to pick a tool and map filters onto
contract-domain fields.
"""

from __future__ import annotations

from typing import Any, Dict

# The five agreed intents.
INTENT_GENERAL = "general"          # broad contract content search (semantic)
INTENT_RISK = "risk"                # structured risk/flag query → unified filter path
INTENT_COUNTERPARTY = "counterparty"  # about a specific counterparty
INTENT_RENEWAL = "renewal"          # renewal / expiry / termination lifecycle
INTENT_CLARIFY = "clarify"          # ambiguous → ask a clarifying question

VALID_INTENTS = (
    INTENT_GENERAL,
    INTENT_RISK,
    INTENT_COUNTERPARTY,
    INTENT_RENEWAL,
    INTENT_CLARIFY,
)

# Which tool each intent routes to.
TOOL_CONTRACT_SEARCH = "contract_search"
TOOL_RISK_SEARCH = "risk_search"
TOOL_CONTRACTS_WHERE = "contracts_where"
TOOL_NONE = "none"

# Risk is computed inside the unified contract_search pipeline (the service
# ranks every candidate set by risk), so the risk intent routes to the same
# contract_search tool rather than a separate risk tool.
INTENT_TO_TOOL: Dict[str, str] = {
    INTENT_GENERAL: TOOL_CONTRACT_SEARCH,
    INTENT_RISK: TOOL_CONTRACT_SEARCH,
    INTENT_COUNTERPARTY: TOOL_CONTRACT_SEARCH,
    INTENT_RENEWAL: TOOL_CONTRACT_SEARCH,
    INTENT_CLARIFY: TOOL_NONE,
}

# Single source of truth for the vague-query clarifying question. The router,
# the agents, the risk planner, and the Streamlit UI all import this so the
# user sees one consistent clarification voice.
DEFAULT_CLARIFICATION = (
    "Could you narrow down which contract, counterparty, or risk area you mean?"
)

# Contract-domain filter fields a query can map onto.
FILTER_CONTRACT_TYPE = "contract_type"
FILTER_DEPARTMENT = "department"
FILTER_COUNTERPARTY = "counterparty_name"
FILTER_DATE_FROM = "date_from"
FILTER_DATE_TO = "date_to"
FILTER_STATUS = "status"
FILTER_EXPIRED = "expired"
FILTER_CONTRACT_ID = "contract_id"

CONTRACT_FILTER_FIELDS = (
    FILTER_CONTRACT_TYPE,
    FILTER_DEPARTMENT,
    FILTER_COUNTERPARTY,
    FILTER_DATE_FROM,
    FILTER_DATE_TO,
    FILTER_STATUS,
    FILTER_EXPIRED,
    FILTER_CONTRACT_ID,
)


def normalize_intent(value: Any) -> str:
    """Coerce an LLM-supplied intent label onto the valid set (safe default)."""
    text = str(value or "").strip().lower()
    for intent in VALID_INTENTS:
        if intent in text:
            return intent
    return INTENT_GENERAL


def infer_intent_from_query(query: str) -> str:
    """Heuristic intent inference used when the router/LLM is unavailable.

    The fallback router is intentionally deterministic, so we keep the rules
    narrow and conservative. Risk questions should stay risk even when the
    LLM provider is missing.
    """
    text = (query or "").strip().lower()
    if not text:
        return INTENT_CLARIFY

    risk_phrases = (
        "risk was not accepted",
        "risk not accepted",
        "not accepted risk",
        "risk accepted = no",
        "needs legal review",
        "risk search",
        "risk-related",
    )
    if any(phrase in text for phrase in risk_phrases):
        return INTENT_RISK

    if "not accepted" in text and "risk" in text:
        return INTENT_RISK

    return INTENT_GENERAL


def default_decision(query: str) -> Dict[str, Any]:
    """Safe-default routing decision used when the LLM is unavailable.

    Defaults to searching the contract corpus (the most common intent).
    """
    return {
        "search": True,
        "intent": INTENT_GENERAL,
        "tool": TOOL_CONTRACT_SEARCH,
        "query": query,
        "filters": {},
        "clarification_question": "",
        "raw": "",
    }
