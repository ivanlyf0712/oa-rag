"""Tests for the unified search pipeline (ticket 01).

Covers: uncapped structured retrieval, semantic dedupe bound, unconditional
risk scoring, relevance-vs-risk ordering, planner integration without a mode
gate, and retirement of gate_and_sort as a result-shaping step.

Run:
    venv/bin/python -m pytest tests/test_unified_search_service.py -v
"""
import json
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.service import (
    RANK_RELEVANCE,
    RANK_RISK,
    SEMANTIC_CONTRACT_LIMIT,
    ContractSearchService,
    UnifiedQueryPlanner,
)


class FakeDB:
    """Minimal sqlite3 database stand-in (2-col rows: id, tags)."""

    def __init__(self, rows):
        self._rows = rows
        self.connection = self
        self._last_rows = rows

    def cursor(self):
        return self

    def execute(self, sql):
        # service tries SELECT id, text, tags first; emulate column error so
        # it falls back to SELECT id, tags — real fakes stay 2-column.
        if "text" in sql:
            raise Exception("no such column: text")
        return self

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeEmbeddings:
    def __init__(self, rows=None):
        self.database = FakeDB(rows or [])


class FakeSearcher:
    def __init__(self, embeddings=None, semantic_results=None):
        self.embeddings = embeddings or FakeEmbeddings()
        self._semantic_results = semantic_results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._semantic_results)

    def _fetch_one_doc(self, doc_id):
        for r in self._semantic_results:
            if r.get("id") == doc_id:
                return r
        return None


class FakePlanner:
    """Stand-in for UnifiedQueryPlanner returning a canned plan."""

    def __init__(self, plan):
        self._plan = plan
        self.queries = []

    def plan(self, query):
        self.queries.append(query)
        return dict(self._plan)


def _tags(**meta):
    return json.dumps(meta)


def _section_rows(n_contracts, chunks_per=2, **meta):
    """Build db rows for n contracts, each with chunks_per sections."""
    rows = []
    for i in range(n_contracts):
        contract_meta = {"contract_id": f"C{i}", "ref_no": f"CCA{i:08d}", **meta}
        for j in range(chunks_per):
            rows.append((f"C{i}-s{j}", _tags(**contract_meta)))
    return rows


# -- structured path: uncapped, contract-level --------------------------

def test_structured_query_returns_all_matches_uncapped():
    # 60 contracts x 2 chunks = 120 sections; type filter matches all of them.
    db_rows = _section_rows(60, contract_type="2")
    searcher = FakeSearcher(embeddings=FakeEmbeddings(rows=db_rows))
    service = ContractSearchService(searcher=searcher)

    # enumeration query -> structured path (free text would stay semantic)
    results = service.search("list all contracts", filters={"contract_type": "2"})

    assert len(results) == 60  # contract-level, not chunk-level, no cap
    assert len(searcher.calls) == 0  # semantic path not used


def test_structured_query_applies_risk_filters():
    metas = [
        {"contract_id": "A", "ref_no": "R1",
         "decoded_fields": {"FlagNeedLegal": {"label": "Yes"}}},
        {"contract_id": "B", "ref_no": "R2",
         "decoded_fields": {"FlagNeedLegal": {"label": "No"}}},
        {"contract_id": "C", "ref_no": "R3", "decoded_fields": {}},
    ]
    db_rows = [(m["ref_no"], _tags(**m)) for m in metas]
    searcher = FakeSearcher(embeddings=FakeEmbeddings(rows=db_rows))
    service = ContractSearchService(searcher=searcher)

    results = service.search(
        "q",
        filters={"risk_filters": [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}]},
    )

    assert [r["metadata"]["ref_no"] for r in results] == ["R1"]


def test_unknown_risk_filter_fields_are_rejected_not_applied():
    db_rows = _section_rows(3, contract_type="1")
    searcher = FakeSearcher(embeddings=FakeEmbeddings(rows=db_rows))
    service = ContractSearchService(searcher=searcher)

    # Bogus field is dropped by validate_filters; with no other filters this
    # is not a structured query at all -> semantic path.
    results = service.search(
        "q", filters={"risk_filters": [{"field": "Bogus", "op": "=", "value": "yes"}]}
    )
    assert results == []  # semantic fake returns nothing; no crash


# -- semantic path: dedupe + bound ---------------------------------------

def test_semantic_search_dedupes_chunks_to_contracts():
    chunks = []
    for i in range(8):  # 8 contracts
        for j in range(3):  # 3 chunks each, best chunk first
            chunks.append({
                "id": f"C{i}-s{j}", "text": f"chunk {j}", "score": 1.0 - j * 0.1,
                "metadata": {"contract_id": f"C{i}", "ref_no": f"R{i}"},
            })
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher)

    results = service.search("liability")

    assert len(results) == 8
    assert [r["id"] for r in results] == [f"C{i}-s0" for i in range(8)]  # best chunk kept


def test_semantic_search_capped_at_contract_limit():
    chunks = [
        {"id": f"C{i}-s0", "text": "t", "score": 1.0,
         "metadata": {"contract_id": f"C{i}", "ref_no": f"R{i}"}}
        for i in range(SEMANTIC_CONTRACT_LIMIT + 20)
    ]
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher)

    results = service.search("broad query")

    assert len(results) == SEMANTIC_CONTRACT_LIMIT


def test_semantic_relevance_order_preserved():
    chunks = [
        {"id": "b", "text": "t", "metadata": {"contract_id": "B", "ref_no": "RB"}},
        {"id": "a", "text": "t", "metadata": {"contract_id": "A", "ref_no": "RA"}},
    ]
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher)

    results = service.search("q", rank_by=RANK_RELEVANCE)
    assert [r["id"] for r in results] == ["b", "a"]  # retrieval order kept


# -- unconditional risk scoring ------------------------------------------

def test_risk_scoring_is_unconditional_on_semantic_results():
    chunks = [
        {"id": "x", "text": "t",
         "metadata": {"contract_id": "X", "ref_no": "RX",
                      "decoded_fields": {"IsRisksAccepted": {"label": "No"},
                                         "Over5M": {"label": "Yes"}}}},
        {"id": "y", "text": "t",
         "metadata": {"contract_id": "Y", "ref_no": "RY", "decoded_fields": {}}},
    ]
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher)

    results = service.search("purchase agreement")  # no risk language at all

    assert len(results) == 2
    for r in results:
        assert "risk_score" in r and "risk_severity" in r
        assert "risk_score" in r["metadata"]
    risky = next(r for r in results if r["metadata"]["ref_no"] == "RX")
    calm = next(r for r in results if r["metadata"]["ref_no"] == "RY")
    assert risky["risk_score"] > 0
    assert risky["matched_signals"]  # signals listed
    assert calm["risk_score"] == 0
    assert calm["risk_severity"] == "low"


def test_no_rows_means_no_scoring_crash():
    searcher = FakeSearcher(semantic_results=[])
    service = ContractSearchService(searcher=searcher)
    assert service.search("nothing here") == []


# -- ranking -------------------------------------------------------------

def test_rank_by_risk_sorts_descending_and_keeps_low_scores():
    chunks = [
        {"id": "low", "text": "t",
         "metadata": {"contract_id": "L", "ref_no": "RL",
                      "decoded_fields": {"Over5M": {"label": "Yes"}}}},  # small score
        {"id": "high", "text": "t",
         "metadata": {"contract_id": "H", "ref_no": "RH",
                      "decoded_fields": {"IsRisksAccepted": {"label": "No"},
                                         "FlagNeedLegal": {"label": "Yes"},
                                         "Over100M": {"label": "Yes"}}}},
        {"id": "zero", "text": "t",
         "metadata": {"contract_id": "Z", "ref_no": "RZ", "decoded_fields": {}}},
    ]
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher)

    results = service.search("q", rank_by=RANK_RISK)

    scores = [r["risk_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # gate_and_sort is retired as result-shaping: zero/low-score rows remain.
    assert [r["id"] for r in results][-1] == "zero"
    assert any(s == 0 for s in scores)


def test_invalid_rank_by_falls_back_to_relevance():
    chunks = [
        {"id": "b", "text": "t", "metadata": {"contract_id": "B", "ref_no": "RB"}},
        {"id": "a", "text": "t", "metadata": {"contract_id": "A", "ref_no": "RA"}},
    ]
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher)

    results = service.search("q", rank_by="banana")
    assert [r["id"] for r in results] == ["b", "a"]


# -- planner integration (mode gate removed) ------------------------------

def test_planner_filters_and_rank_hint_applied():
    plan = {
        "filters": [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}],
        "risk_intent": True,
        "rank_hint": RANK_RISK,
        "explanation": "explicit risk language",
    }
    metas = [
        {"contract_id": "A", "ref_no": "R1",
         "decoded_fields": {"FlagNeedLegal": {"label": "Yes"},
                            "IsRisksAccepted": {"label": "No"}}},
        {"contract_id": "B", "ref_no": "R2",
         "decoded_fields": {"FlagNeedLegal": {"label": "No"}}},
    ]
    db_rows = [(m["ref_no"], _tags(**m)) for m in metas]
    searcher = FakeSearcher(embeddings=FakeEmbeddings(rows=db_rows))
    planner = FakePlanner(plan)
    service = ContractSearchService(searcher=searcher, planner=planner)

    results = service.search("contracts needing legal review")

    assert planner.queries == ["contracts needing legal review"]
    assert [r["metadata"]["ref_no"] for r in results] == ["R1"]
    assert service.last_plan["risk_intent"] is True


def test_planner_never_gates_general_queries():
    # A general plan (no filters, relevance hint) must not suppress results.
    plan = {"filters": [], "risk_intent": False,
            "rank_hint": RANK_RELEVANCE, "explanation": "general"}
    chunks = [{"id": "x", "text": "t",
               "metadata": {"contract_id": "X", "ref_no": "RX"}}]
    searcher = FakeSearcher(semantic_results=chunks)
    service = ContractSearchService(searcher=searcher, planner=FakePlanner(plan))

    results = service.search("purchase agreements with Alpha")

    assert len(results) == 1
    assert results[0]["risk_score"] == 0  # scored anyway


def test_explicit_filters_skip_planner():
    chunks = [{"id": "x", "text": "t",
               "metadata": {"contract_id": "X", "ref_no": "RX", "contract_type": "2"}}]
    db_rows = [("x", _tags(contract_id="X", ref_no="RX", contract_type="2"))]
    searcher = FakeSearcher(embeddings=FakeEmbeddings(rows=db_rows),
                            semantic_results=chunks)
    planner = FakePlanner({"filters": [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}],
                           "risk_intent": True, "rank_hint": RANK_RISK, "explanation": ""})
    service = ContractSearchService(searcher=searcher, planner=planner)

    results = service.search("q", filters={"contract_type": "2"})

    assert planner.queries == []  # planner not consulted
    assert service.last_plan is None
    assert len(results) == 1


# -- UnifiedQueryPlanner over RiskPlanner ----------------------------------

def test_unified_planner_keyword_risk_intent():
    planner = UnifiedQueryPlanner(risk_planner=_StubRiskPlanner({
        "mode": "risky_contracts", "confidence": 0.85,
        "filters": [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}],
        "explanation": "explicit risk language",
    }))
    plan = planner.plan("show risky contracts needing legal review")
    assert plan["risk_intent"] is True
    assert plan["rank_hint"] == RANK_RISK
    assert plan["filters"] == [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}]


def test_unified_planner_clarify_becomes_general_hint():
    planner = UnifiedQueryPlanner(risk_planner=_StubRiskPlanner({
        "mode": "clarify", "confidence": 0.4, "filters": [],
        "explanation": "ambiguous",
        "clarification_question": "risk or general?",
    }))
    plan = planner.plan("contracts")
    assert plan["risk_intent"] is False
    assert plan["rank_hint"] == RANK_RELEVANCE
    assert plan["filters"] == []


class _StubRiskPlanner:
    """Mimics RiskPlanner.plan without LLM/keywords."""

    def __init__(self, plan):
        self._plan = plan

    def plan(self, query):
        return dict(self._plan)


# -- exact ref path --------------------------------------------------------

def test_exact_ref_lookup_still_deterministic_and_scored():
    db_rows = [("s1", _tags(ref_no="CCA20250096", contract_id="C1",
                            decoded_fields={"Over5M": {"label": "Yes"}}))]
    rows = [{"id": "s1", "text": "body",
             "metadata": {"ref_no": "CCA20250096", "contract_id": "C1",
                          "decoded_fields": {"Over5M": {"label": "Yes"}}}}]
    searcher = FakeSearcher(embeddings=FakeEmbeddings(rows=db_rows),
                            semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    results = service.search("CCA20250096")

    assert len(searcher.calls) == 0  # semantic bypassed
    assert len(results) == 1
    assert results[0]["metadata"]["ref_no"] == "CCA20250096"
    assert results[0]["risk_score"] > 0  # unconditional scoring on exact path too
