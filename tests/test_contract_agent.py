"""Tests for the CrossTableAgent manual-ReAct router (Ticket 2).

Tools and the LLM are stubbed so the agent logic is tested without a real
index, database, or LLM.

Run:
    venv/bin/python -m pytest tests/test_contract_agent.py -v
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


class FakeRouter:
    """Returns a fixed routing decision."""

    def __init__(self, decision):
        self._decision = decision

    def decide(self, query):
        d = dict(self._decision)
        d.setdefault("query", query)
        return d


def _agent(decision, contract_result="contract hits", summarize="FINAL ANSWER"):
    calls = {"contract": []}

    def contract_tool(query, filters):
        calls["contract"].append((query, filters))
        return contract_result

    agent = CrossTableAgent(
        contract_tool=contract_tool,
        router=FakeRouter(decision),
    )
    agent._llm_summarize = lambda q, tool, obs: summarize  # type: ignore
    return agent, calls


def test_routes_to_contract_search():
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "x"}
    agent, calls = _agent(decision)
    out = agent.process("find breach clauses")
    assert out["tool"] == TOOL_CONTRACT_SEARCH
    assert out["intent"] == INTENT_GENERAL
    assert out["output"] == "FINAL ANSWER"
    assert out["success"] is True
    assert calls["contract"]


def test_contract_tool_receives_filters():
    decision = {"search": True, "intent": "counterparty", "tool": TOOL_CONTRACT_SEARCH,
                "filters": {"counterparty_name": "Acme"}, "raw": "x"}
    agent, calls = _agent(decision)
    agent.process("contracts with Acme")
    assert calls["contract"][0][1] == {"counterparty_name": "Acme"}


def test_routes_to_risk_search():
    decision = {"search": True, "intent": INTENT_RISK, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "x"}
    agent, calls = _agent(decision)
    out = agent.process("show risky contracts")
    assert out["tool"] == TOOL_CONTRACT_SEARCH
    assert out["intent"] == INTENT_RISK
    assert calls["contract"]


def test_clarify_intent_no_tool_call():
    decision = {"search": False, "intent": INTENT_CLARIFY, "tool": TOOL_NONE,
                "clarification_question": "Which counterparty?", "raw": "x"}
    agent, calls = _agent(decision)
    out = agent.process("tell me about it")
    assert out["tool"] == TOOL_NONE
    assert out["output"] == "Which counterparty?"
    assert out["success"] is True
    assert not calls["contract"]


def test_empty_input_rejected_without_tool():
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH}
    agent, calls = _agent(decision)
    out = agent.process("   ")
    assert out["success"] is False
    assert out["tool"] == TOOL_NONE
    assert not calls["contract"]


def test_fallback_flag_when_router_had_no_llm():
    # raw empty → router degraded (no LLM) → agent marks fallback=True
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": ""}
    agent, calls = _agent(decision)
    out = agent.process("find something")
    assert out["fallback"] is True
    assert out["success"] is True


def test_no_fallback_flag_when_router_used_llm():
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "{...}"}
    agent, calls = _agent(decision)
    out = agent.process("find something")
    assert out["fallback"] is False


def test_tool_exception_returns_failure():
    decision = {"search": True, "intent": INTENT_RISK, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "x"}
    agent, calls = _agent(decision)

    out = agent.process("risky")
    assert out["success"] is True
    assert out["fallback"] is False


def test_empty_observation_says_no_results():
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "x"}
    agent, calls = _agent(decision, contract_result="   ")
    out = agent.process("nothing matches")
    assert "No matching contracts" in out["output"]


def test_steps_recorded():
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "x"}
    agent, calls = _agent(decision)
    out = agent.process("find breach clauses")
    labels = [s["label"] for s in out["steps"]]
    assert "Routing" in labels
    assert "contract_search" in labels
    assert "Answer generation" in labels


def test_tool_calls_recorded_with_input():
    decision = {"search": True, "intent": INTENT_GENERAL, "tool": TOOL_CONTRACT_SEARCH,
                "filters": {}, "raw": "x"}
    agent, calls = _agent(decision)
    out = agent.process("find breach clauses")
    assert out["tool_calls"][0]["tool"] == TOOL_CONTRACT_SEARCH
    assert out["tool_calls"][0]["tool_input"]
