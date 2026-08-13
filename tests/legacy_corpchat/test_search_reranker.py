#!/usr/bin/env python3
"""
Reranker tests for ticket 04 — multilingual reranker on the verified base.

Tests the Searcher.search() seam with use_rerank=True and asserts results are
at least as relevant as use_rerank=False (base-only). Uses a FakeReranker for
deterministic, no-download testing, plus a real-model integration test that
skips when the model is not cached locally.

Run:
    conda run -n ocr pytest tests/test_search_reranker.py -v
"""
import inspect
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
    Reranker,
    DEFAULT_RERANKER_MODEL,
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
            match_text = f"{title}\n---\n{text}"
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
    tmp = tmp_path_factory.mktemp("corpchat_rerank")
    idx = _build_test_index(tmp)
    yield idx


@pytest.fixture(scope="session")
def embeddings(test_index):
    embeddings = txtai.Embeddings()
    embeddings.load(test_index)
    return embeddings


# ── FakeReranker: deterministic, no model download ──────────────
class FakeReranker:
    """
    Deterministic reranker that simulates a Chinese-capable cross-encoder.

    Scores each (query, document) pair by keyword overlap, mimicking what a
    good Chinese reranker does: documents containing more query keywords rank
    higher. This lets us test the full rerank wiring without a 1GB+ download.
    """

    def __init__(self, top_n: int = 20):
        self.enabled = True
        self.model = None
        self.model_name = "fake-chinese-reranker"
        self.top_n = top_n

    def rerank(self, query: str, results: list) -> list:
        if not self.enabled or not results:
            return results

        if len(results) <= self.top_n:
            to_rerank = results
            rest = []
        else:
            to_rerank = results[: self.top_n]
            rest = results[self.top_n :]

        # Extract meaningful keywords from the query (Chinese chars + words)
        query_keywords = set()
        for char in query:
            if "\u4e00" <= char <= "\u9fff":
                query_keywords.add(char)
        # Also add multi-char terms
        terms = query.split()
        for term in terms:
            if len(term) > 1:
                query_keywords.add(term)

        for item in to_rerank:
            text = item.get("text", "")
            overlap = sum(1 for kw in query_keywords if kw in text)
            item["rerank_score"] = float(overlap)

        to_rerank.sort(key=lambda x: float(x.get("rerank_score", 0)), reverse=True)
        return to_rerank + rest


# ── Relevance scoring helper ─────────────────────────────────────
def _relevance_score(text: str, keywords: list) -> float:
    """Score a document by how many query keywords it contains."""
    return float(sum(1 for kw in keywords if kw in text))


def _total_relevance(results: list, keywords: list, k: int = 3) -> float:
    """Sum of relevance scores for top-k results."""
    return sum(_relevance_score(r.get("text", ""), keywords) for r in results[:k])


# ── Unit tests (no model download) ───────────────────────────────
def test_default_reranker_model_is_chinese_capable():
    """The default reranker model must be Chinese-capable, not ms-marco-MiniLM."""
    assert "ms-marco" not in DEFAULT_RERANKER_MODEL.lower(), (
        f"Default reranker is still English-only: {DEFAULT_RERANKER_MODEL}"
    )
    assert "bge" in DEFAULT_RERANKER_MODEL.lower() or \
           "multilingual" in DEFAULT_RERANKER_MODEL.lower() or \
           "m3" in DEFAULT_RERANKER_MODEL.lower(), (
        f"Default reranker is not Chinese-capable: {DEFAULT_RERANKER_MODEL}"
    )


def test_reranker_uses_default_model():
    """Reranker() with no args must use the Chinese-capable default model."""
    r = Reranker()
    assert r.model_name == DEFAULT_RERANKER_MODEL
    assert r.model_name != "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_search_use_rerank_defaults_to_true():
    """Searcher.search() must default use_rerank=True (enabled by default)."""
    sig = inspect.signature(Searcher.search)
    assert sig.parameters["use_rerank"].default is True, (
        "use_rerank must default to True — reranking is enabled by default"
    )


def test_reranker_enabled_when_sentence_transformers_available():
    """Reranker.enabled must be True when sentence_transformers is installed."""
    r = Reranker()
    # sentence_transformers is in requirements.txt, so should be available
    assert r.enabled, "Reranker should be enabled when sentence_transformers is installed"


# ── Integration tests with FakeReranker (no model download) ──────
def test_rerank_logistics_improves_or_matches_base(embeddings):
    """物流報價 方案: reranked results must be at least as relevant as base-only."""
    query = "物流報價 方案"
    keywords = ["物流", "報價"]
    searcher = Searcher(embeddings, reranker=FakeReranker())

    base = searcher.search(query, mode="hybrid", limit=5, expand=False,
                           graph_expand=0, use_rerank=False)
    reranked = searcher.search(query, mode="hybrid", limit=5, expand=False,
                               graph_expand=0, use_rerank=True)

    assert base and reranked, "No results returned"

    # Reranked top-3 must still contain the logistics message
    reranked_top = [r.get("text", "") for r in reranked[:3]]
    assert any("物流" in t and "報價" in t for t in reranked_top), (
        f"Reranked results lost the logistics message: {reranked_top}"
    )

    # Reranked total relevance >= base total relevance (improve-or-match)
    base_rel = _total_relevance(base, keywords)
    reranked_rel = _total_relevance(reranked, keywords)
    assert reranked_rel >= base_rel, (
        f"Reranking degraded relevance: base={base_rel}, reranked={reranked_rel}"
    )


def test_rerank_investment_improves_or_matches_base(embeddings):
    """投資美國債券跟藍籌股: reranked results must be at least as relevant."""
    query = "投資美國債券跟藍籌股"
    keywords = ["債券", "藍籌"]
    searcher = Searcher(embeddings, reranker=FakeReranker())

    base = searcher.search(query, mode="hybrid", limit=5, expand=False,
                           graph_expand=0, use_rerank=False)
    reranked = searcher.search(query, mode="hybrid", limit=5, expand=False,
                                graph_expand=0, use_rerank=True)

    assert base and reranked, "No results returned"

    # Reranked top-3 must still contain the investment message
    reranked_top = [r.get("text", "") for r in reranked[:3]]
    assert any("債券" in t and "藍籌" in t for t in reranked_top), (
        f"Reranked results lost the investment message: {reranked_top}"
    )

    # Reranked total relevance >= base total relevance (improve-or-match)
    base_rel = _total_relevance(base, keywords)
    reranked_rel = _total_relevance(reranked, keywords)
    assert reranked_rel >= base_rel, (
        f"Reranking degraded relevance: base={base_rel}, reranked={reranked_rel}"
    )


def test_rerank_label_filter_still_scopes(embeddings):
    """Label filter must still scope correctly with reranking enabled."""
    searcher = Searcher(embeddings, reranker=FakeReranker())
    results = searcher.search(
        "投資", mode="hybrid", limit=10, expand=False, graph_expand=0,
        use_rerank=True, label_filter="investment_opportunity"
    )
    assert results, "No results returned"
    labels = [r.get("metadata", {}).get("label") for r in results]
    assert all(l == "investment_opportunity" for l in labels), (
        f"Label filter leaked other labels: {labels}"
    )


def test_rerank_bare_label_does_not_rank_all_label_docs(embeddings):
    """Bare-label search with reranking must not rank all docs of that label."""
    searcher = Searcher(embeddings, reranker=FakeReranker())
    results = searcher.search(
        "product_inquiry", mode="hybrid", limit=10, expand=False,
        graph_expand=0, use_rerank=True
    )
    assert results, "No results returned"
    labels = [r.get("metadata", {}).get("label") for r in results]
    pi_count = sum(1 for l in labels if l == "product_inquiry")
    assert pi_count < len(results), (
        f"Bare label search with reranking ranked all {pi_count}/{len(results)} docs as product_inquiry"
    )


# ── Real model integration test (skips if model not cached) ──────
def _is_model_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is cached locally."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = "models--" + model_name.replace("/", "--")
    model_path = os.path.join(cache_dir, model_dir)
    return os.path.isdir(model_path) and any(
        os.path.isfile(os.path.join(model_path, "snapshots", s, "config.json"))
        for s in os.listdir(os.path.join(model_path, "snapshots"))
        if os.path.isdir(os.path.join(model_path, "snapshots", s))
    ) if os.path.isdir(os.path.join(model_path, "snapshots")) else False


@pytest.mark.skipif(
    not _is_model_cached(DEFAULT_RERANKER_MODEL),
    reason=f"{DEFAULT_RERANKER_MODEL} not cached locally — skipping real-model test"
)
def test_real_reranker_improves_chinese_queries(embeddings):
    """Real Chinese-capable reranker improves-or-matches base on Chinese queries."""
    reranker = Reranker()
    assert reranker.enabled, "Reranker not enabled"
    searcher = Searcher(embeddings, reranker=reranker)

    test_cases = [
        ("物流報價 方案", ["物流", "報價"]),
        ("投資美國債券跟藍籌股", ["債券", "藍籌"]),
    ]

    for query, keywords in test_cases:
        base = searcher.search(query, mode="hybrid", limit=5, expand=False,
                               graph_expand=0, use_rerank=False)
        reranked = searcher.search(query, mode="hybrid", limit=5, expand=False,
                                   graph_expand=0, use_rerank=True)

        assert base and reranked, f"No results for query: {query}"

        # Reranked top-3 must still contain the relevant message
        reranked_top = [r.get("text", "") for r in reranked[:3]]
        assert any(all(kw in t for kw in keywords) for t in reranked_top), (
            f"Real reranker lost relevant message for '{query}': {reranked_top}"
        )

        # Improve-or-match: reranked relevance >= base relevance
        base_rel = _total_relevance(base, keywords)
        reranked_rel = _total_relevance(reranked, keywords)
        assert reranked_rel >= base_rel, (
            f"Real reranker degraded relevance for '{query}': "
            f"base={base_rel}, reranked={reranked_rel}"
        )