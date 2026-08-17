"""Tests for ticket 05: contracts_where SQL tool (Phase 2).

Ported in spirit from corpchat test_tools_expansion.py: rule path, LLM
path (mocked client), invalid SQL rejected, index-scan fallback. Uses a
real in-memory SQLite sections table so json_extract behaves exactly as
in production.

Run:
    venv/bin/python -m pytest tests/test_contracts_where.py -v
"""
import json
import os
import sqlite3
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.service import ContractSearchService
from apps.search.result_store import clear_results, snapshot_results
from apps.search.where_sql import (
    _condition_to_sql,
    _validate_sql,
    condition_to_sql,
    enumeration_remainder,
)
from apps.search_cli import build_where_tool


# ── fakes: real in-memory SQLite behind the searcher seam ──────────

class _SQLiteEmbeddings:
    def __init__(self, conn):
        self.database = type("DB", (), {"connection": conn})()


class _FakeSearcher:
    def __init__(self, conn, semantic_results=None):
        self.embeddings = _SQLiteEmbeddings(conn)
        self._semantic = semantic_results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._semantic)


class _FakeLLM:
    """LiteLLMClient stand-in returning a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def chat(self, messages, **kwargs):
        self.requests.append(messages)
        return self.reply


def _meta(ref, **over):
    m = {"ref_no": ref, "title": ref + " contract"}
    m.update(over)
    return m


def _seed():
    """Three contracts; CCA001 has two chunks (dedupe check)."""
    rows = [
        ("CCA001#0", "chunk one alpha", _meta(
            "CCA001", amount=6_000_000, contract_end_date="2026-06-30",
            department="IT", status_label="active",
            decoded_fields={"Over5M": {"raw": 1, "label": "yes"},
                            "FlagNeedLegal": {"raw": 1, "label": "yes"},
                            "IsRisksAccepted": {"raw": 0, "label": "no"}})),
        ("CCA001#1", "chunk two alpha", _meta(
            "CCA001", amount=6_000_000, contract_end_date="2026-06-30",
            department="IT", status_label="active")),
        ("CCA002#0", "beta services", _meta(
            "CCA002", amount=2_000_000, contract_end_date="2028-01-15",
            department="Finance", status_label="completed",
            decoded_fields={"Over5M": {"raw": 0, "label": "no"},
                            "FlagNeedLegal": {"raw": 0, "label": "no"},
                            "IsRisksAccepted": {"raw": 1, "label": "yes"}})),
        ("CCA003#0", "gamma supply", _meta(
            "CCA003", amount=500_000, contract_end_date="2025-03-01",
            department="Legal", status_label="active",
            decoded_fields={"IsRisksAccepted": {"raw": 1, "label": "yes"}})),
    ]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sections (indexid INTEGER PRIMARY KEY, id TEXT, text TEXT, tags TEXT)")
    for i, (doc_id, text, meta) in enumerate(rows):
        conn.execute("INSERT INTO sections VALUES (?, ?, ?, ?)",
                     (i, doc_id, text, json.dumps(meta)))
    conn.commit()
    return conn


def _service(semantic_results=None):
    conn = _seed()
    searcher = _FakeSearcher(conn, semantic_results=semantic_results)
    return ContractSearchService(searcher=searcher), searcher


def _refs(rows):
    return sorted((r.get("metadata") or {}).get("ref_no") for r in rows)


# ── SQL translation units ───────────────────────────────────────────

def test_validate_sql_guardrails():
    assert _validate_sql("SELECT id FROM sections") == \
        "SELECT id FROM sections LIMIT 5000"
    assert _validate_sql("  select id, tags from sections limit 9; ") == \
        "select id, tags from sections limit 9"
    assert _validate_sql("DROP TABLE sections") is None
    assert _validate_sql("SELECT id FROM sections; DELETE FROM sections") is None
    assert _validate_sql("SELECT id FROM other_table") is None
    assert _validate_sql("SELECT id FROM sections WHERE 1=1 --") is None
    assert _validate_sql("") is None
    assert _validate_sql(None) is None


def test_condition_to_sql_rule_shapes():
    assert "> 5000000" in _condition_to_sql("contracts over HK$5M")
    assert ">= 10000000" in _condition_to_sql("at least HK$10M")
    assert "< 500000" in _condition_to_sql("under 500k")
    assert "Over5M" in _condition_to_sql("contracts over 5m")
    assert "contract_end_date" in _condition_to_sql("ending before 2027")
    assert "contract_start_date" in _condition_to_sql("starting after 2024")
    assert "FlagNeedLegal" in _condition_to_sql("contracts needing legal review")
    sql = _condition_to_sql("risk not accepted")
    assert "IsRisksAccepted" in sql and "= 'no'" in sql
    assert "status_label" in _condition_to_sql("completed contracts")
    assert _condition_to_sql("quantum entanglement") is None


def test_enumeration_remainder_strips_boilerplate():
    assert enumeration_remainder("list all contracts") == ""
    assert enumeration_remainder("show all") == ""
    assert enumeration_remainder("list all contracts with risk not accepted") == \
        "risk not accepted"
    assert condition_to_sql("list all", allow_llm=False) == \
        "SELECT id, text, tags FROM sections LIMIT 5000"


# ── service.search_where: rule path ─────────────────────────────────

def test_rule_amount_comparison():
    svc, _ = _service()
    assert _refs(svc.search_where("contracts over HK$5M", allow_llm=False)) == ["CCA001"]


def test_rule_threshold_flag():
    svc, _ = _service()
    rows = svc.search_where("list all contracts over 5m", allow_llm=False)
    assert _refs(rows) == ["CCA001"]


def test_rule_date_bound():
    svc, _ = _service()
    assert _refs(svc.search_where("contracts ending before 2027", allow_llm=False)) == \
        ["CCA001", "CCA003"]


def test_rule_coded_flag():
    svc, _ = _service()
    assert _refs(svc.search_where("contracts needing legal review", allow_llm=False)) == \
        ["CCA001"]


def test_rule_risk_not_accepted_enumeration():
    svc, _ = _service()
    rows = svc.search_where("list all contracts with risk not accepted", allow_llm=False)
    assert _refs(rows) == ["CCA001"]


def test_bare_list_all_returns_every_contract():
    svc, searcher = _service()
    rows = svc.search_where("list all contracts", allow_llm=False)
    assert _refs(rows) == ["CCA001", "CCA002", "CCA003"]
    assert searcher.calls == []  # structured path, never vector search


def test_empty_condition_returns_every_contract():
    svc, _ = _service()
    assert _refs(svc.search_where("", allow_llm=False)) == ["CCA001", "CCA002", "CCA003"]


def test_dedupe_and_risk_scoring():
    svc, _ = _service()
    rows = svc.search_where("list all contracts", allow_llm=False)
    assert len(rows) == 3  # CCA001's two chunks collapse to one row
    for r in rows:
        meta = r["metadata"]
        assert "risk_score" in meta and "risk_severity" in meta
    by_ref = {(r["metadata"] or {}).get("ref_no"): r for r in rows}
    assert by_ref["CCA001"]["metadata"]["risk_score"] > \
        by_ref["CCA003"]["metadata"]["risk_score"]


# ── LLM fallback + rejection + index scan ───────────────────────────

def test_llm_translation_fallback_mocked():
    svc, _ = _service()
    client = _FakeLLM("SELECT id, text, tags FROM sections "
                      "WHERE json_extract(tags, '$.department') = 'IT'")
    rows = svc.search_where("contracts from the IT department", llm_client=client)
    assert client.requests  # LLM was consulted
    assert _refs(rows) == ["CCA001"]


def test_invalid_llm_sql_rejected_then_index_scan():
    semantic = [{"id": "S1", "text": "fallback hit", "score": 0.9,
                 "metadata": {"ref_no": "CCA009"}}]
    svc, searcher = _service(semantic_results=semantic)
    rows = svc.search_where("gibberish condition", llm_client=_FakeLLM("DROP TABLE sections"))
    assert searcher.calls  # fell back to the index scan
    assert _refs(rows) == ["CCA009"]


def test_sql_execution_failure_falls_back():
    svc, searcher = _service(semantic_results=[{"id": "S1", "text": "x", "score": 0.5,
                                                "metadata": {"ref_no": "CCA010"}}])
    svc._searcher.embeddings.database = None  # simulate broken DB handle
    rows = svc.search_where("risk not accepted", allow_llm=False)
    assert searcher.calls  # index-scan fallback, no exception
    assert _refs(rows) == ["CCA010"]


# ── tool wiring ─────────────────────────────────────────────────────

def test_where_tool_stashes_rows_and_formats_observation():
    svc, _ = _service()
    tool = build_where_tool(service=svc)
    clear_results()
    obs = tool("list all contracts with risk not accepted")
    assert "CCA001" in obs
    snap = snapshot_results()
    assert _refs(snap["rows"]) == ["CCA001"]
    assert snap["query"] == "list all contracts with risk not accepted"
    clear_results()


def test_langchain_tools_register_contracts_where():
    from apps.search.langchain_agent import build_langchain_tools
    calls = []
    tools = build_langchain_tools(
        contract_tool=lambda q, f: "obs",
        where_tool=lambda c: calls.append(c) or "where-obs",
    )
    names = [t.name for t in tools]
    assert "contracts_where" in names and "contract_search" in names
    where = next(t for t in tools if t.name == "contracts_where")
    assert where.invoke({"condition": "over HK$5M"}) == "where-obs"
    assert calls == ["over HK$5M"]


def test_cross_table_agent_routes_enumeration_to_where():
    from apps.search.agent import CrossTableAgent

    class _Router:
        def decide(self, query):
            return {"intent": "general", "tool": "contract_search",
                    "query": query, "filters": {}, "raw": {}}

    seen = []
    agent = CrossTableAgent(
        contract_tool=lambda q, f: "semantic:" + q,
        where_tool=lambda c: seen.append(c) or "structured:" + c,
        router=_Router(),
    )
    result = agent.process("list all contracts with risk not accepted")
    assert result["tool"] == "contracts_where"
    assert seen == ["list all contracts with risk not accepted"]
    assert result["success"] is True

# ── contracts_aggregate: SQL builder (whitelist, validated) ──────
from apps.search.where_sql import aggregate_sql


def test_aggregate_sql_count_by_department():
    sql = aggregate_sql("count", "department", "")
    assert sql is not None
    low = sql.lower()
    assert low.startswith("select")
    assert "count(*)" in low
    assert "json_extract(tags, '$.department')" in low
    assert "group by" in low
    assert _validate_sql(sql) is not None


def test_aggregate_sql_sum_amount_no_group():
    sql = aggregate_sql("sum_amount", "", "")
    assert sql is not None
    low = sql.lower()
    # Dedupe subquery: per-contract amount via MAX over the chunk group.
    assert "max(cast(json_extract(tags, '$.amount') as real))" in low
    assert "sum(amount)" in low
    # No outer GROUP BY (single overall figure); inner dedupe groups by ck.
    assert "group by ck" in low
    assert "group by grp" not in low


def test_aggregate_sql_avg_amount_by_year():
    sql = aggregate_sql("avg_amount", "year", "")
    assert sql is not None
    low = sql.lower()
    assert "avg(amount)" in low  # outer aggregate over deduped per-contract rows
    assert "strftime('%Y'" in sql  # 4-digit year group expression
    assert "group by grp" in low


def test_aggregate_sql_injects_translated_condition():
    # "over 5m" is a threshold-flag rule -> Over5M label WHERE clause injected.
    sql = aggregate_sql("count", "department", "contracts over 5m")
    assert sql is not None
    low = sql.lower()
    assert "where" in low
    assert "over5m" in low  # threshold flag, not a raw $.amount comparison
    assert "group by" in low
    assert _validate_sql(sql) is not None


def test_aggregate_sql_rejects_unknown_metric_and_group():
    assert aggregate_sql("median", "department", "") is None
    assert aggregate_sql("count", "salary", "") is None
    assert aggregate_sql("count'; DROP TABLE sections; --", "department", "") is None



# ── service.aggregate: execution + rendering ─────────────────────

def test_aggregate_count_by_department():
    svc, _ = _service()
    out = svc.aggregate("count", "department", "")
    # 3 distinct departments, one row each (CCA001/2/3).
    assert "IT" in out and "Finance" in out and "Legal" in out
    assert out.count("\n") >= 3  # header + at least 3 group rows
    # Each department has exactly 1 contract.
    assert "1" in out


def test_aggregate_sum_amount_overall():
    svc, _ = _service()
    out = svc.aggregate("sum_amount", "", "")
    # 6_000_000 + 2_000_000 + 500_000 = 8_500_000 total (no GROUP BY).
    assert "8500000" in out or "8.5" in out or "8,500,000" in out


def test_aggregate_sum_amount_filtered_by_condition():
    svc, _ = _service()
    # Only CCA001 is Over5M -> total 6_000_000.
    out = svc.aggregate("sum_amount", "", "contracts over 5m")
    assert "6000000" in out or "6.0" in out or "6,000,000" in out


def test_aggregate_unknown_metric_returns_message_not_raise():
    svc, _ = _service()

def test_rank_by_amount_sorts_descending():
    svc, _ = _service()
    # "list all"-style structured query returns all contracts; rank by amount.
    rows = svc.search_where("", allow_llm=False, rank_by="amount")
    amounts = [ (r.get("metadata") or {}).get("amount") for r in rows ]
    assert amounts == sorted(amounts, reverse=True)
    assert amounts[0] == 6_000_000  # CCA001 highest


# ── contract_detail: single-contract drill-down ──────────────────

def test_contract_detail_merges_chunks_and_formats():
    svc, _ = _service()
    out = svc.contract_detail("CCA001")
    assert "CCA001" in out
    assert "IT" in out                 # department
    assert "active" in out.lower()     # status
    assert "HK$6.0M" in out or "6000000" in out  # amount (deduped, not 12M)
    # risk decoded from decoded_fields -> non-zero for CCA001
    assert "risk" in out.lower()


def test_contract_detail_case_insensitive_and_unknown():
    svc, _ = _service()
    assert "CCA002" in svc.contract_detail("cca002")  # case-insensitive
    out = svc.contract_detail("NOPE999")
    assert "no contract" in out.lower() or "not found" in out.lower()


# ── contracts_compare: side-by-side across refs ──────────────────

def test_contracts_compare_two_refs_side_by_side():
    svc, _ = _service()
    out = svc.contracts_compare(["CCA001", "CCA002"])
    low = out.lower()
    assert "field" in low            # header column
    assert "CCA001" in out and "CCA002" in out  # both columns present
    # Field rows for the compared dimensions.
    for field in ("department", "status", "amount", "risk"):
        assert field in low
    # Distinct values surface: IT vs Finance.
    assert "IT" in out and "Finance" in out


def test_contracts_compare_handles_unknown_ref():
    svc, _ = _service()
    out = svc.contracts_compare(["CCA001", "NOPE999"])
    assert "CCA001" in out
    assert "NOPE999" in out  # column still rendered
    assert "not found" in out.lower()  # unknown ref marked in its column


def test_contracts_compare_requires_at_least_two_refs():
    svc, _ = _service()
    out = svc.contracts_compare(["CCA001"])
    assert "two" in out.lower() or "2" in out

    out = svc.aggregate("median", "department", "")
    assert isinstance(out, str)
    assert "unsupported" in out.lower() or "unknown" in out.lower()

def test_aggregate_sql_is_read_only_single_statement():
    sql = aggregate_sql("sum_amount", "counterparty_name", "")
    assert sql is not None
    assert ";" not in sql
    for bad in ("insert", "update", "delete", "drop", "alter", "create"):
        assert bad not in sql.lower()

