#!/usr/bin/env python3
"""
Expansion-aware tests for ticket 02 — LLM query expansion on the verified base.

Tests the Searcher.search() seam with expand=True (using a mocked QueryExpander)
and asserts results are at least as relevant as expand=False (base-only).

Run:
    conda run -n ocr pytest tests/test_search_expansion.py -v
"""
import json
import os
import sys

import pytest
import txtai

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import (
    Searcher,
    QueryExpander,
    ORIGINAL_QUERY_WEIGHT,
    LLM_SEMANTIC_QUERY_WEIGHT,
    LLM_KEYWORD_QUERY_WEIGHT,
)
from apps.corpchat.gen_fake_msg import CONVERSATION_TEMPLATES, CONTACTS


# ── Fixture: deterministic in-memory index ──────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"


def _build_test_index(tmp_path):
    """Build a tiny txtai index from conversation templates."""
    docs = []
    for conv in CONVERSATION_TEMPLATES:
        label = conv["label"]
        init_name = CONTACTS[conv["initiator"]]["name"]
        resp_name = CONTACTS[conv["responder"]]["name"]
        for i, (speaker_idx, text) in enumerate(conv["turns"]):
            speaker_name = CONTACTS[speaker_idx]["name"]
            doc_id = f"{label}_{i}"
            title = f"{speaker_name} ({label})"
            # Match surface: content + curated title only
            match_text = f"{title}\n---\n{text}"
            # Structured metadata (filter/display/LLM-context only)
            tags = {
                "label": label,
                "customer_name": speaker_name,
                "company": CONTACTS[speaker_idx]["company"],
                "send_time": "2026-01-01T00:00:00",
                "external_userid": speaker_name,
                "servicer_userid": resp_name if speaker_name == init_name else init_name,
                "msgid": doc_id,
                "origin": "3" if speaker_name == init_name else "5",
                "chunk_index": i,
            }
            docs.append((doc_id, match_text, json.dumps(tags, default=str)))

    embeddings = txtai.Embeddings(
        {
            "path": EMBEDDING_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
        }
    )
    embeddings.index(docs)

    idx_path = os.path.join(tmp_path, "test_idx")
    embeddings.save(idx_path)
    return idx_path


@pytest.fixture(scope="session")
def test_index(tmp_path_factory):
    """Session-scoped deterministic index."""
    tmp = tmp_path_factory.mktemp("corpchat")
    idx = _build_test_index(tmp)
    yield idx


@pytest.fixture(scope="session")
def embeddings(test_index):
    """Session-scoped txtai embeddings loaded from the test index."""
    embeddings = txtai.Embeddings()
    embeddings.load(test_index)
    return embeddings


# ── FakeExpander: deterministic, no live API ────────────────────
class FakeExpander:
    """Deterministic QueryExpander that returns pre-computed expansions."""

    def __init__(self, expansions: dict):
        self._expansions = expansions

    def expand(self, query: str, use_cache: bool = True):
        return self._expansions.get(query, [(query, ORIGINAL_QUERY_WEIGHT)])


# ── QueryExpander unit tests (mocked LLM, no live API) ─────────
class _MockLLMExpander(QueryExpander):
    """QueryExpander with _call_llm stubbed to return canned responses."""

    def __init__(self, semantic_response: str = "", keyword_response: str = ""):
        super().__init__(api_base="http://mock", api_key="mock-key")
        self._semantic_response = semantic_response
        self._keyword_response = keyword_response

    def _call_llm(self, messages, max_tokens=200):
        # Route based on the system prompt content
        system = messages[0]["content"] if messages else ""
        if "semantic" in system.lower():
            return self._semantic_response
        if "keyword" in system.lower():
            return self._keyword_response
        return ""


def test_query_expander_generates_semantic_and_keyword_queries():
    """QueryExpander must produce original + semantic + keyword queries with correct weights."""
    expander = _MockLLMExpander(
        semantic_response="物流系統報價方案",
        keyword_response="物流 報價\n報價單",
    )
    result = expander.expand("物流報價 方案", use_cache=False)

    # Original query with weight 0.5
    assert (("物流報價 方案", ORIGINAL_QUERY_WEIGHT)) in result
    # Semantic rephrase with weight 1.3
    assert (("物流系統報價方案", LLM_SEMANTIC_QUERY_WEIGHT)) in result
    # Keyword expansions with weight 1.0
    assert (("物流 報價", LLM_KEYWORD_QUERY_WEIGHT)) in result
    assert (("報價單", LLM_KEYWORD_QUERY_WEIGHT)) in result


def test_query_expander_falls_back_to_original_on_llm_failure():
    """QueryExpander must degrade gracefully when LLM calls fail."""
    expander = _MockLLMExpander(semantic_response="", keyword_response="")
    result = expander.expand("投資美國債券跟藍籌股", use_cache=False)

    # Only the original query should be returned
    assert result == [("投資美國債券跟藍籌股", ORIGINAL_QUERY_WEIGHT)]


def test_query_expander_deduplicates_queries():
    """QueryExpander must not add duplicate queries."""
    expander = _MockLLMExpander(
        semantic_response="投資美國債券跟藍籌股",  # same as original
        keyword_response="投資美國債券跟藍籌股",  # same as original
    )
    result = expander.expand("投資美國債券跟藍籌股", use_cache=False)

    # Only the original query should be returned (deduplicated)
    assert result == [("投資美國債券跟藍籌股", ORIGINAL_QUERY_WEIGHT)]


# ── Expansion-aware tests ───────────────────────────────────────
def test_expansion_logistics_quotation_matches_base(embeddings):
    """物流報價 方案 with expansion must still surface the logistics message."""
    query = "物流報價 方案"
    fake = FakeExpander({
        query: [
            (query, ORIGINAL_QUERY_WEIGHT),
            ("物流系統報價方案", LLM_SEMANTIC_QUERY_WEIGHT),
            ("物流 報價", LLM_KEYWORD_QUERY_WEIGHT),
        ]
    })
    searcher = Searcher(embeddings, expander=fake)

    base_results = searcher.search(query, mode="hybrid", limit=5, expand=False, graph_expand=0, use_rerank=False)
    exp_results = searcher.search(query, mode="hybrid", limit=5, expand=True, graph_expand=0, use_rerank=False)

    assert base_results, "Base search returned no results"
    assert exp_results, "Expanded search returned no results"

    base_top = [r.get("text", "") for r in base_results[:3]]
    exp_top = [r.get("text", "") for r in exp_results[:3]]

    # Base must find the logistics message
    assert any("物流" in t and "報價" in t for t in base_top), f"Base failed: {base_top}"
    # Expansion must also find it (at least as relevant)
    assert any("物流" in t and "報價" in t for t in exp_top), f"Expansion degraded: {exp_top}"


def test_expansion_investment_bond_bluechip_matches_base(embeddings):
    """投資美國債券跟藍籌股 with expansion must still surface the investment message."""
    query = "投資美國債券跟藍籌股"
    fake = FakeExpander({
        query: [
            (query, ORIGINAL_QUERY_WEIGHT),
            ("投資美國債券藍籌股", LLM_SEMANTIC_QUERY_WEIGHT),
            ("美國債券 藍籌股", LLM_KEYWORD_QUERY_WEIGHT),
        ]
    })
    searcher = Searcher(embeddings, expander=fake)

    base_results = searcher.search(query, mode="hybrid", limit=5, expand=False, graph_expand=0, use_rerank=False)
    exp_results = searcher.search(query, mode="hybrid", limit=5, expand=True, graph_expand=0, use_rerank=False)

    assert base_results, "Base search returned no results"
    assert exp_results, "Expanded search returned no results"

    base_top = [r.get("text", "") for r in base_results[:3]]
    exp_top = [r.get("text", "") for r in exp_results[:3]]

    # Base must find the investment message
    assert any("債券" in t and "藍籌" in t for t in base_top), f"Base failed: {base_top}"
    # Expansion must also find it (at least as relevant)
    assert any("債券" in t and "藍籌" in t for t in exp_top), f"Expansion degraded: {exp_top}"


def test_expansion_label_filter_still_scopes(embeddings):
    """Label filter must still scope correctly with expansion enabled."""
    query = "投資"
    fake = FakeExpander({
        query: [
            (query, ORIGINAL_QUERY_WEIGHT),
            ("投資方案", LLM_SEMANTIC_QUERY_WEIGHT),
            ("投資 債券", LLM_KEYWORD_QUERY_WEIGHT),
        ]
    })
    searcher = Searcher(embeddings, expander=fake)

    results = searcher.search(
        query, mode="hybrid", limit=10, expand=True, graph_expand=0,
        use_rerank=False, label_filter="investment_opportunity"
    )
    assert results, "No results returned"
    labels = [r.get("metadata", {}).get("label") for r in results]
    assert all(l == "investment_opportunity" for l in labels), (
        f"Label filter leaked other labels: {labels}"
    )