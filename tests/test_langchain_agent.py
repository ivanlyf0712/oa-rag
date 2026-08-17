"""Regression gate for the LangChain tool-calling agent.

Asserts only *external* behaviour of apps.search.langchain_agent.LangChainAgent:
which tool a query routes to, which intent is chosen, clarification/fallback
handling, the public result shape, and deterministic degradation when the LLM
is unavailable. It does NOT depend on a live index, database, or LLM — the
tests drive the real LangGraph ReAct loop through a scripted BaseChatModel.

Run:
    venv/bin/python -m pytest tests/test_langchain_agent.py -v
"""
import os
import sys
from typing import Any, Dict, List

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

NL = chr(10)  # newline, avoids heredoc escaping issues
from apps.search.intents import (
    INTENT_CLARIFY,
    INTENT_GENERAL,
    INTENT_RISK,
    TOOL_CONTRACT_SEARCH,
    TOOL_NONE,
)
from apps.search.langchain_agent import (
    AgentConfigError,
    LangChainAgent,
    build_default_llm,
    build_langchain_tools,
)

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseChatModel = object  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")


# ── Scripted fake LangChain chat model ───────────────────────────
class ScriptedLLM(BaseChatModel):
    """Scripted BaseChatModel that drives the real LangGraph ReAct loop.

    Script values:
      ("tool", name, args) → respond to a matching query with that tool call
      ("text", content)    → respond with plain content (clarify / answer)

    Unmatched queries default to a plain contract_search call forwarding the
    query.  Once a ToolMessage is present (second pass after tool execution),
    respond with the final natural-language answer.
    """

    script: Dict[str, Any] = {}
    bound_tools: List[Any] = []
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    @staticmethod
    def _result(message):
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return self._result(AIMessage(content="Scripted answer based on tool results."))
        query = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                query = str(m.content)
                break
        for needle, spec in self.script.items():
            if needle in query:
                if spec[0] == "text":
                    return self._result(AIMessage(content=spec[1]))
                _, name, args = spec
                return self._result(AIMessage(content="", tool_calls=[
                    {"name": name, "args": dict(args), "id": "call_%d" % self.calls}]))
        return self._result(AIMessage(content="", tool_calls=[
            {"name": TOOL_CONTRACT_SEARCH,
             "args": {"query": query, "filters": {}},
             "id": "call_%d" % self.calls}]))

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


class RecordingTools:
    def __init__(self):
        self.contract_calls: List[Dict[str, Any]] = []
        self.risk_calls: List[Dict[str, Any]] = []

    def contract(self, query, filters=None):
        self.contract_calls.append({"query": query, "filters": filters})
        return "CONTRACT RESULTS"

    def risk(self, query, filters=None):
        self.risk_calls.append({"query": query, "filters": filters})
        return "RISK RESULTS"


ROUTING_SCRIPT = {
    # Order matters: first matching needle wins.
    "contract ID 12345": ("tool", TOOL_CONTRACT_SEARCH,
                          {"query": "show contract ID 12345", "filters": {"contract_id": "12345"}}),
    "not accepted": ("tool", TOOL_CONTRACT_SEARCH, {"query": "risk not accepted"}),
    "high risk": ("tool", TOOL_CONTRACT_SEARCH, {"query": "high risk contracts"}),
    "risk": ("tool", TOOL_CONTRACT_SEARCH, {"query": "risk not accepted"}),
    "liability": ("tool", TOOL_CONTRACT_SEARCH, {"query": "unlimited liability", "filters": {}}),
    "breach": ("tool", TOOL_CONTRACT_SEARCH, {"query": "breach clause", "filters": {}}),
    "Acme": ("tool", TOOL_CONTRACT_SEARCH,
             {"query": "Acme contracts", "filters": {"counterparty_name": "Acme"}}),
    "renewal": ("tool", TOOL_CONTRACT_SEARCH, {"query": "renewal contracts", "filters": {}}),
    "completed": ("tool", TOOL_CONTRACT_SEARCH,
                  {"query": "show completed contracts", "filters": {"status": "completed"}}),
    "expired": ("tool", TOOL_CONTRACT_SEARCH,
                {"query": "which contracts have expired", "filters": {"expired": True}}),
    "vague": ("text", "Which contract or counterparty do you mean?"),
}

GOLDEN_CASES = [
    ("which contracts mention unlimited liability", TOOL_CONTRACT_SEARCH, INTENT_GENERAL),
    ("show breach clauses", TOOL_CONTRACT_SEARCH, INTENT_GENERAL),
    ("show contracts where risk was not accepted", TOOL_CONTRACT_SEARCH, INTENT_RISK),
    ("risk not accepted contracts", TOOL_CONTRACT_SEARCH, INTENT_RISK),
    ("what contracts do we have with Acme", TOOL_CONTRACT_SEARCH, INTENT_GENERAL),
    ("which contracts are up for renewal", TOOL_CONTRACT_SEARCH, INTENT_GENERAL),
    ("vague tell me about it", TOOL_NONE, INTENT_CLARIFY),
]


@pytest.fixture()
def tools():
    return RecordingTools()


@pytest.fixture()
def agent(tools):
    return LangChainAgent(
        contract_tool=tools.contract,
        llm=ScriptedLLM(script=ROUTING_SCRIPT),
        synthesize=lambda q, t, o: "ANSWER[%s]: %s" % (t, o),
    )


# ── Golden routing regression ────────────────────────────────────
@pytest.mark.parametrize("query,expected_tool,expected_intent", GOLDEN_CASES)
def test_routing_golden_matrix(agent, query, expected_tool, expected_intent):
    out = agent.process(query)
    assert out["tool"] == expected_tool
    assert out["intent"] == expected_intent


# ── Tool selection exclusivity ───────────────────────────────────
def test_contract_query_invokes_only_contract_tool(agent, tools):
    agent.process("which contracts mention unlimited liability")
    assert tools.contract_calls and not tools.risk_calls


def test_risk_query_invokes_unified_contract_tool(agent, tools):
    # Candidate 2: risk queries route to the unified contract_search tool (the
    # service applies risk filters/ranking); the risk intent is preserved.
    out = agent.process("show contracts where risk was not accepted")
    assert out["tool"] == TOOL_CONTRACT_SEARCH
    assert out["intent"] == INTENT_RISK


# ── Fast-route comparative gate ─────────────────────────────────
@pytest.mark.parametrize(
    "query",
    [
        "compare risk-not-accepted contracts across departments",
        "risk not accepted vs accepted counts",
        "why was risk not accepted",
        "how many risk not accepted contracts by month",
    ],
)
def test_fast_route_yields_comparative_risk(agent, query):
    # Analytical/comparative risk queries are not certain single-tool calls, so
    # the fast path must yield (None) and let the ReAct engine handle them.
    assert agent._fast_route(query) is None


def test_fast_route_keeps_pure_risk(agent):
    # A pure risk retrieval stays on the fast path (guards against over-yield).
    decision = agent._fast_route("risk not accepted contracts")
    assert decision is not None
    assert decision["tool"] == TOOL_CONTRACT_SEARCH
    assert decision["intent"] == INTENT_RISK


def test_fast_route_keeps_bare_ref(agent):
    # The bare ref-number branch is untouched by the comparative gate.
    decision = agent._fast_route("CCA20250096")
    assert decision is not None
    assert decision["tool"] == TOOL_CONTRACT_SEARCH
    assert decision["intent"] == INTENT_GENERAL


def test_normal_search_is_default_tool(agent, tools):
    out = agent.process("show breach clauses")
    assert out["tool"] == TOOL_CONTRACT_SEARCH
    assert tools.contract_calls and not tools.risk_calls


def test_clarify_calls_no_tool(agent, tools):
    out = agent.process("vague tell me about it")
    assert out["tool"] == TOOL_NONE
    assert out["clarify"] is True
    assert not tools.contract_calls


# ── Tool-call args forwarded ─────────────────────────────────────
def test_tool_call_filters_forwarded(agent, tools):
    agent.process("what contracts do we have with Acme")
    assert tools.contract_calls[0]["filters"].get("counterparty_name") == "Acme"


def test_status_completed_filter_forwarded(agent, tools):
    out = agent.process("show completed contracts")
    assert tools.contract_calls[0]["filters"].get("status") == "completed"
    assert out["tool_calls"][0]["filters"].get("status") == "completed"


def test_contract_id_filter_forwarded(agent, tools):
    out = agent.process("show contract ID 12345")
    assert tools.contract_calls[0]["filters"].get("contract_id") == "12345"
    assert out["tool_calls"][0]["filters"].get("contract_id") == "12345"


def test_expired_filter_forwarded(agent, tools):
    out = agent.process("which contracts have expired")
    assert tools.contract_calls[0]["filters"].get("expired") is True
    assert out["tool_calls"][0]["filters"].get("expired") is True


# ── Public result shape / metadata ───────────────────────────────
def test_result_exposes_public_seam_keys(agent):
    out = agent.process("show breach clauses")
    for key in ("output", "intent", "tool", "tool_calls", "steps",
                "success", "fallback", "clarify", "observation"):
        assert key in out, "missing public seam key: %s" % key
    assert isinstance(out["tool_calls"], list)
    assert isinstance(out["steps"], list)


def test_tool_provenance_recorded(agent):
    out = agent.process("show breach clauses")
    assert out["tool_calls"][0]["tool"] == TOOL_CONTRACT_SEARCH
    assert out["tool_calls"][0]["tool_input"]
    assert out["observation"] == "CONTRACT RESULTS"


# ── default synthesize uses the LLM, not a prompt echo ───────────
def test_default_synthesize_returns_llm_summary_not_prompt(tools):
    """_default_synthesize must call the LLM and return ITS summary,
    never echo the raw synthesis prompt or the raw evidence dump."""
    llm_summary = "These are mostly renewal and expiration contracts, several not yet risk-accepted."
    synth_llm = ScriptedLLM(
        script={"Summarize the overall contract search results": ("text", llm_summary)})
    a = LangChainAgent(contract_tool=tools.contract, llm=synth_llm)
    evidence = NL.join(["Over5M: yes", "IsRisksAccepted: no", "1. [ABC] Renewal", "2. [XYZ] Contract Expiration Reminder"])
    out = a._default_synthesize("show contracts where risk was not accepted", TOOL_CONTRACT_SEARCH, evidence)
    assert out == llm_summary
    assert "overall contract search results" not in out
    assert "Do not list record IDs" not in out
    assert "User query:" not in out


def test_default_synthesize_falls_back_gracefully_when_llm_down(tools):
    class BrokenLLM:
        def bind_tools(self, t, **kw):
            return self

        def invoke(self, messages, **kw):
            raise RuntimeError("provider down")

    a = LangChainAgent(contract_tool=tools.contract, llm=BrokenLLM())
    evidence = NL.join(["1. [ABC] Renewal", "2. [XYZ] Expiration"])
    out = a._default_synthesize("find completed contracts", TOOL_CONTRACT_SEARCH, evidence)
    assert "overall contract search results" not in out
    assert "User query:" not in out
    assert "[ABC]" not in out
    assert "unavailable" in out.lower()
    assert "2" in out


def test_default_synthesize_empty_observation():
    a = LangChainAgent()
    assert a._default_synthesize("q", TOOL_CONTRACT_SEARCH, "   ") == "No matching contracts were found."


# ── build_langchain_tools wiring: aggregate tool + rank_by ───────

def _tool_names(built):
    return {getattr(t, "name", "") for t in built}


def test_build_tools_includes_aggregate_when_provided():
    built = build_langchain_tools(
        contract_tool=lambda q, filters=None, rank_by=None: "C",
        where_tool=lambda c: "W",
        aggregate_tool=lambda metric, group_by="", condition="": "A",
    )
    assert "contracts_aggregate" in _tool_names(built)
    assert "contract_search" in _tool_names(built)
    assert "contracts_where" in _tool_names(built)


def test_build_tools_omits_aggregate_when_not_provided():
    built = build_langchain_tools(
        contract_tool=lambda q, filters=None, rank_by=None: "C",
        where_tool=lambda c: "W",
    )
    assert "contracts_aggregate" not in _tool_names(built)


def test_contract_search_forwards_rank_by():
    seen = {}

    def contract(query, filters=None, rank_by=None):
        seen["rank_by"] = rank_by
        return "OK"

    built = build_langchain_tools(contract_tool=contract, where_tool=lambda c: "W")
    cs = next(t for t in built if getattr(t, "name", "") == "contract_search")
    cs.func(query="big contracts", rank_by="amount")
    assert seen["rank_by"] == "amount"
    cs.func(query="big contracts")  # rank_by omitted -> None
    assert seen["rank_by"] is None


def test_aggregate_tool_invoked_with_three_args():
    seen = {}

    def aggregate(metric, group_by="", condition=""):
        seen["args"] = (metric, group_by, condition)
        return "TABLE"

    built = build_langchain_tools(
        contract_tool=lambda q, filters=None, rank_by=None: "C",
        where_tool=lambda c: "W",
        aggregate_tool=aggregate,
    )
    agg = next(t for t in built if getattr(t, "name", "") == "contracts_aggregate")
    assert agg.func(metric="sum_amount", group_by="department", condition="over 5m") == "TABLE"
    assert seen["args"] == ("sum_amount", "department", "over 5m")


def test_build_tools_includes_detail_and_compare_when_provided():
    built = build_langchain_tools(
        contract_tool=lambda q, filters=None: "C",
        where_tool=lambda c: "W",
        aggregate_tool=lambda metric, group_by="", condition="": "A",
        detail_tool=lambda ref: "D",
        compare_tool=lambda refs: "CMP",
    )
    names = _tool_names(built)
    assert "contract_detail" in names
    assert "contracts_compare" in names


def test_build_tools_omits_detail_and_compare_when_not_provided():
    built = build_langchain_tools(
        contract_tool=lambda q, filters=None: "C",
        where_tool=lambda c: "W",
    )
    names = _tool_names(built)
    assert "contract_detail" not in names
    assert "contracts_compare" not in names


def test_detail_and_compare_tools_forward_args():
    seen = {}

    def detail(ref):
        seen["ref"] = ref
        return "DETAIL"

    def compare(refs):
        seen["refs"] = refs
        return "COMPARE"

    built = build_langchain_tools(
        contract_tool=lambda q, filters=None: "C",
        where_tool=lambda c: "W",
        detail_tool=detail,
        compare_tool=compare,
    )
    tm = {getattr(t, "name", ""): t for t in built}
    assert tm["contract_detail"].func(ref="CCA001") == "DETAIL"
    assert seen["ref"] == "CCA001"
    assert tm["contracts_compare"].func(refs=["CCA001", "CCA002"]) == "COMPARE"
    assert seen["refs"] == ["CCA001", "CCA002"]

