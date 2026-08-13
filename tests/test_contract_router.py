"""Tests for the contract-domain SearchRouter.

The router decides whether to search, classifies into one of the contract
intents, and maps filters onto contract-domain fields. LLM calls are stubbed so
no live API is needed.

Run:
    venv/bin/python -m pytest tests/test_contract_router.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.intents import (
    INTENT_CLARIFY,
    INTENT_COUNTERPARTY,
    INTENT_GENERAL,
    INTENT_RENEWAL,
    INTENT_RISK,
    TOOL_CONTRACT_SEARCH,
    TOOL_NONE,
    TOOL_RISK_SEARCH,
)
from apps.search.router import SearchRouter


def _router_returning(payload: str) -> SearchRouter:
    router = SearchRouter()
    router._call_llm = lambda messages, max_tokens=300: payload  # type: ignore
    return router


# ── search gate ──────────────────────────────────────────────────
def test_greeting_does_not_search():
    router = _router_returning('{"search": false, "intent": "general", "query": "hi"}')
    d = router.decide("hello")
    assert d["search"] is False
    assert d["intent"] == INTENT_GENERAL


def test_contract_content_question_searches_contract_tool():
    router = _router_returning(
        '{"search": true, "intent": "general", "query": "breach liability clause"}'
    )
    d = router.decide("which contracts mention unlimited liability?")
    assert d["search"] is True
    assert d["intent"] == INTENT_GENERAL
    assert d["tool"] == TOOL_CONTRACT_SEARCH
    assert d["query"] == "breach liability clause"


# ── intent model ─────────────────────────────────────────────────
def test_risk_intent_routes_to_risk_tool():
    router = _router_returning(
        '{"search": true, "intent": "risk", "query": "risk not accepted"}'
    )
    d = router.decide("show contracts where risk was not accepted")
    assert d["intent"] == INTENT_RISK
    assert d["tool"] == TOOL_RISK_SEARCH


def test_counterparty_intent_maps_filter():
    router = _router_returning(
        '{"search": true, "intent": "counterparty", "query": "Acme contracts", '
        '"filters": {"counterparty_name": "Acme"}}'
    )
    d = router.decide("what contracts do we have with Acme?")
    assert d["intent"] == INTENT_COUNTERPARTY
    assert d["filters"]["counterparty_name"] == "Acme"
    assert d["tool"] == TOOL_CONTRACT_SEARCH


def test_renewal_intent():
    router = _router_returning('{"search": true, "intent": "renewal", "query": "renewal contracts"}')
    d = router.decide("which contracts are up for renewal?")
    assert d["intent"] == INTENT_RENEWAL


def test_clarify_intent_disables_search():
    router = _router_returning(
        '{"search": true, "intent": "clarify", "clarification_question": "Which counterparty?"}'
    )
    d = router.decide("tell me about it")
    assert d["intent"] == INTENT_CLARIFY
    assert d["search"] is False
    assert d["tool"] == TOOL_NONE
    assert d["clarification_question"] == "Which counterparty?"


# ── filter mapping ───────────────────────────────────────────────
def test_date_range_filters_kept_and_unknown_dropped():
    router = _router_returning(
        '{"search": true, "intent": "general", "query": "q", '
        '"filters": {"date_from": "2024-01-01", "date_to": "2024-12-31", "bogus": "x"}}'
    )
    d = router.decide("contracts signed in 2024")
    assert d["filters"]["date_from"] == "2024-01-01"
    assert d["filters"]["date_to"] == "2024-12-31"
    assert "bogus" not in d["filters"]


def test_status_filter_mapped():
    router = _router_returning(
        '{"search": true, "intent": "general", "query": "q", '
        '"filters": {"status": "completed"}}'
    )
    d = router.decide("show completed contracts")
    assert d["filters"]["status"] == "completed"


def test_expired_filter_mapped():
    router = _router_returning(
        '{"search": true, "intent": "renewal", "query": "q", '
        '"filters": {"expired": true}}'
    )
    d = router.decide("which contracts have expired")
    assert d["filters"]["expired"] == "True"


def test_contract_id_filter_mapped():
    router = _router_returning(
        '{"search": true, "intent": "general", "query": "q", '
        '"filters": {"contract_id": "CCA20250096"}}'
    )
    d = router.decide("lookup CCA20250096")
    assert d["filters"]["contract_id"] == "CCA20250096"


def test_unknown_intent_label_normalizes_to_general():
    router = _router_returning('{"search": true, "intent": "banana", "query": "q"}')
    d = router.decide("anything")
    assert d["intent"] == INTENT_GENERAL


# ── safe degradation ─────────────────────────────────────────────
def test_malformed_json_falls_back_to_safe_default():
    router = _router_returning("not json at all")
    d = router.decide("find contract X")
    assert d["search"] is True
    assert d["intent"] == INTENT_GENERAL
    assert d["tool"] == TOOL_CONTRACT_SEARCH
    assert d["query"] == "find contract X"


def test_llm_exception_degrades_safely():
    router = SearchRouter()
    def _boom(messages, max_tokens=300):
        raise RuntimeError("network down")
    router._call_llm = _boom  # type: ignore
    d = router.decide("find contract X")
    assert d["search"] is True
    assert d["tool"] == TOOL_CONTRACT_SEARCH


def test_empty_query_returns_clarify_no_search():
    router = SearchRouter()
    d = router.decide("   ")
    assert d["search"] is False
    assert d["intent"] == INTENT_CLARIFY
    assert d["tool"] == TOOL_NONE


def test_decision_is_cached():
    router = _router_returning('{"search": true, "intent": "risk", "query": "risk"}')
    calls = {"n": 0}
    real = router._call_llm
    def counting(messages, max_tokens=300):
        calls["n"] += 1
        return real(messages, max_tokens)
    router._call_llm = counting  # type: ignore
    router.decide("risk?")
    router.decide("risk?")
    assert calls["n"] == 1
