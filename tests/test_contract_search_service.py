"""Tests for apps.search.service.ContractSearchService.

Uses in-memory fakes for the Searcher and txtai database so no live index,
LLM, or SQLite file is required.

Run:
    venv/bin/python -m pytest tests/test_contract_search_service.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.service import ContractSearchService, _looks_like_ref_no, _normalize_status


class FakeDB:
    """Minimal sqlite3 database stand-in."""

    def __init__(self, rows):
        self._rows = rows
        self.connection = self

    def cursor(self):
        return self

    def execute(self, _sql):
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


# -- helper unit tests -------------------------------------------------

def test_looks_like_ref_no():
    assert _looks_like_ref_no("CCA20250096")
    assert _looks_like_ref_no("CKTEST080604")
    assert not _looks_like_ref_no("contract about liability")
    assert not _looks_like_ref_no("")


def test_normalize_status_synonyms():
    assert _normalize_status("done") == "completed"
    assert _normalize_status("Finished") == "completed"
    assert _normalize_status("completed") == "completed"
    assert _normalize_status(None) is None


# -- semantic search path ----------------------------------------------

def test_service_routes_semantic_search_and_post_filters():
    rows = [
        {"id": "a", "text": "t1", "metadata": {"counterparty_name": "Alpha Corp", "contract_type": "2"}},
        {"id": "b", "text": "t2", "metadata": {"counterparty_name": "Beta Ltd", "contract_type": "2"}},
    ]
    searcher = FakeSearcher(semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    results = service.search("liability", filters={"contract_type": "2", "counterparty_name": "Alpha"})

    assert len(results) == 1
    assert results[0]["id"] == "a"
    call = searcher.calls[0]
    assert call["mode"] == "hybrid"
    assert call["expand"] is False
    assert call["use_rerank"] is False
    assert call["label_filter"] == "2"


def test_service_status_filter_normalizes_synonyms():
    rows = [
        {"id": "x", "text": "t", "metadata": {"status_label": "completed", "contract_type": "1"}},
        {"id": "y", "text": "t", "metadata": {"status": "in_review", "contract_type": "1"}},
    ]
    searcher = FakeSearcher(semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    results = service.search("contracts", filters={"status": "done"})
    assert [r["id"] for r in results] == ["x"]


def test_service_returns_empty_when_filters_exclude_all():
    rows = [
        {"id": "x", "text": "t", "metadata": {"counterparty_name": "Alpha", "contract_type": "1"}},
    ]
    searcher = FakeSearcher(semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    assert service.search("anything", filters={"counterparty_name": "Beta"}) == []


def test_service_limit_is_applied():
    rows = [{"id": str(i), "text": "t", "metadata": {"contract_type": "1"}} for i in range(20)]
    searcher = FakeSearcher(semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    assert len(service.search("q", limit=5)) == 5


# -- exact-reference path ----------------------------------------------

def test_service_exact_ref_lookup():
    rows = [
        {"id": "ref1", "text": "body", "metadata": {"ref_no": "CCA20250096", "contract_type": "2"}},
    ]
    db_rows = [("ref1", '{"ref_no": "CCA20250096", "contract_type": "2"}')]
    embeddings = FakeEmbeddings(rows=db_rows)
    searcher = FakeSearcher(embeddings=embeddings, semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    results = service.search("CCA20250096")

    assert len(results) == 1
    assert results[0]["metadata"]["ref_no"] == "CCA20250096"
    assert len(searcher.calls) == 0  # semantic path bypassed


def test_service_exact_ref_respects_status_filter():
    rows = [
        {"id": "ref1", "text": "body", "metadata": {"ref_no": "CCA20250096", "status_label": "completed"}},
    ]
    db_rows = [("ref1", '{"ref_no": "CCA20250096", "status_label": "completed"}')]
    embeddings = FakeEmbeddings(rows=db_rows)
    searcher = FakeSearcher(embeddings=embeddings, semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    assert service.search("CCA20250096", filters={"status": "in_review"}) == []


def test_service_exact_ref_respects_contract_type():
    rows = [
        {"id": "ref1", "text": "body", "metadata": {"ref_no": "CCA20250096", "contract_type": "2"}},
    ]
    db_rows = [("ref1", '{"ref_no": "CCA20250096", "contract_type": "2"}')]
    embeddings = FakeEmbeddings(rows=db_rows)
    searcher = FakeSearcher(embeddings=embeddings, semantic_results=rows)
    service = ContractSearchService(searcher=searcher)

    assert service.search("CCA20250096", filters={"contract_type": "1"}) == []


# -- CLI tool integration ----------------------------------------------

def test_build_contract_tool_uses_service():
    from apps.search_cli import build_contract_tool

    rows = [
        {"id": "a", "text": "body", "metadata": {"counterparty_name": "Alpha", "contract_type": "2", "ref_no": "R1"}},
    ]
    searcher = FakeSearcher(semantic_results=rows)
    tool = build_contract_tool(embeddings=None, searcher=searcher)
    out = tool("liability", {"contract_type": "2"})

    assert "Alpha" in out
    assert "R1" in out
