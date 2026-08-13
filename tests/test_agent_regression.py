"""Regression gate for the public contract-agent seam (Ticket 3).

Asserts only *external* behavior of the agent: which tool a query routes to,
which intent is chosen, and that LLM-down behavior degrades deterministically.
It does NOT depend on internal implementation, a live index, database, or LLM.

The stale CorpChat-specific agent tests (tests/test_agent.py, which imports
the removed apps.corpchat package) are superseded by this gate.

Run:
    venv/bin/python -m pytest tests/test_agent_regression.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.agent import CrossTableAgent
from apps.search.intents import (
    INTENT_CLARIFY,
    INTENT_GENERAL,
    INTENT_RISK,
    TOOL_CONTRACT_SEARCH,
    TOOL_NONE,
)


# ── Deterministic stubs ──────────────────────────────────────────
class ScriptedRouter:
    """Router whose decisions come from a query→decision script (no LLM)."""

    def __init__(self, script, default):
        self._script = script
        self._default = default

    def decide(self, query):
        for needle, decision in self._script.items():
            if needle in query:
                d = dict(decision)
                d.setdefault("query", query)
                return d
        d = dict(self._default)
        d.setdefault("query", query)
        return d


class RecordingTools:
    """Fake tool that records calls and returns a canned observation."""

    def __init__(self):
        self.contract_calls = []

    def contract(self, query, filters):
        self.contract_calls.append({"query": query, "filters": filters})
        return "CONTRACT RESULTS"


def _decide(intent, tool, search=True, filters=None, clarification="", raw="llm"):
    return {
        "search": search,
        "intent": intent,
        "tool": tool,
        "filters": filters or {},
        "clarification_question": clarification,
        "raw": raw,
    }


# ── Golden routing matrix (external behavior contract) ──────────
# needle-in-query → expected (tool, intent)
ROUTING_SCRIPT = {
    "liability": _decide(INTENT_GENERAL, TOOL_CONTRACT_SEARCH),
    "breach": _decide(INTENT_GENERAL, TOOL_CONTRACT_SEARCH),
    "risk": _decide(INTENT_RISK, TOOL_CONTRACT_SEARCH),
    "not accepted": _decide(INTENT_RISK, TOOL_CONTRACT_SEARCH),
    "Acme": _decide("counterparty", TOOL_CONTRACT_SEARCH,
                     filters={"counterparty_name": "Acme"}),
    "renewal": _decide("renewal", TOOL_CONTRACT_SEARCH),
    "vague": _decide(INTENT_CLARIFY, TOOL_NONE, search=False,
                     clarification="Which contract or counterparty?"),
}
DEFAULT_DECISION = _decide(INTENT_GENERAL, TOOL_CONTRACT_SEARCH)

GOLDEN_CASES = [
    ("which contracts mention unlimited liability", TOOL_CONTRACT_SEARCH, INTENT_GENERAL),
    ("show breach clauses", TOOL_CONTRACT_SEARCH, INTENT_GENERAL),
    ("show contracts where risk was not accepted", TOOL_CONTRACT_SEARCH, INTENT_RISK),
    ("risk not accepted contracts", TOOL_CONTRACT_SEARCH, INTENT_RISK),
    ("what contracts do we have with Acme", TOOL_CONTRACT_SEARCH, "counterparty"),
    ("which contracts are up for renewal", TOOL_CONTRACT_SEARCH, "renewal"),
    ("vague tell me about it", TOOL_NONE, INTENT_CLARIFY),
]


@pytest.fixture()
def tools():
    return RecordingTools()


@pytest.fixture()
def agent(tools):
    a = CrossTableAgent(
        contract_tool=tools.contract,
        router=ScriptedRouter(ROUTING_SCRIPT, DEFAULT_DECISION),
    )
    # deterministic answer synthesis (no LLM)
    a._llm_summarize = lambda q, tool, obs: f"ANSWER[{tool}]: {obs}"  # type: ignore
    return a


# ── Golden routing regression ────────────────────────────────────
@pytest.mark.parametrize("query,expected_tool,expected_intent", GOLDEN_CASES)
def test_routing_golden_matrix(agent, tools, query, expected_tool, expected_intent):
    out = agent.process(query)
    assert out["tool"] == expected_tool
    assert out["intent"] == expected_intent


# ── Tool selection exclusivity ───────────────────────────────────
def test_risk_queries_call_contract_tool(agent, tools):
    agent.process("show contracts where risk was not accepted")
    assert tools.contract_calls


def test_contract_queries_call_contract_tool(agent, tools):
    agent.process("which contracts mention unlimited liability")
    assert tools.contract_calls


def test_counterparty_filter_is_forwarded(agent, tools):
    agent.process("what contracts do we have with Acme")
    assert tools.contract_calls[0]["filters"].get("counterparty_name") == "Acme"


def test_clarify_calls_no_tool(agent, tools):
    out = agent.process("vague tell me about it")
    assert out["tool"] == TOOL_NONE
    assert not tools.contract_calls


# ── Public seam result shape ─────────────────────────────────────
def test_result_exposes_public_seam_keys(agent):
    out = agent.process("show breach clauses")
    for key in ("output", "intent", "tool", "tool_calls", "steps", "success", "fallback"):
        assert key in out, f"missing public seam key: {key}"
    assert isinstance(out["tool_calls"], list)
    assert isinstance(out["steps"], list)


# ── LLM-down deterministic degradation ──────────────────────────
def _agent_with_no_llm_router(tools):
    """Router that degrades (no raw LLM output) → safe default to contract search."""
    from apps.search.router import SearchRouter
    router = SearchRouter()
    router._call_llm = lambda messages, max_tokens=300: ""  # LLM unavailable
    a = CrossTableAgent(
        contract_tool=tools.contract,
        router=router,
    )
    a._llm_summarize = lambda q, tool, obs: f"ANSWER[{tool}]: {obs}"  # type: ignore
    return a


def test_llm_down_defaults_to_contract_search_deterministically(tools):
    agent = _agent_with_no_llm_router(tools)
    # Even a risk-flavoured query, with the LLM down, must deterministically
    # fall back to the safe default (contract search), never raise.
    out1 = agent.process("show contracts where risk was not accepted")
    out2 = agent.process("show contracts where risk was not accepted")
    assert out1["tool"] == TOOL_CONTRACT_SEARCH
    assert out1["fallback"] is True
    assert out1["success"] is True
    # deterministic: same input → same routing when LLM is down
    assert out1["tool"] == out2["tool"]
    assert out1["intent"] == out2["intent"]


def test_llm_down_never_raises_and_marks_fallback(tools):
    agent = _agent_with_no_llm_router(tools)
    for q in ("liability", "risk", "Acme", "renewal"):
        out = agent.process(q)
        assert out["success"] is True
        assert out["fallback"] is True
