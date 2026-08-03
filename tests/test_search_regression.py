#!/usr/bin/env python3
"""
Regression tests for ticket 01 — Chinese-capable hybrid base.

Tests the Searcher.search() seam with a deterministic in-memory index
built from the conversation templates. Uses the production embedding
model (BAAI/bge-m3) so tests exercise exactly what runs in production.

Run:
    conda run -n ocr pytest tests/test_search_regression.py -v
"""
import json
import os
import sys
import tempfile

import pytest
import txtai

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import (
    Searcher,
    QueryExpander,
    Reranker,
    DEFAULT_INDEX_PATH,
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
def searcher(test_index):
    embeddings = txtai.Embeddings()
    embeddings.load(test_index)
    return Searcher(embeddings)


# ── Regression assertions ───────────────────────────────────────
def test_logistics_quotation_returns_relevant_message(searcher):
    """物流報價 方案 must return the logistics quotation message, not just any 方案 doc."""
    results = searcher.search("物流報價 方案", mode="hybrid", limit=5, expand=False, graph_expand=0, use_rerank=False)
    assert results, "No results returned"
    top_texts = [r.get("text", "") for r in results[:3]]
    # The top result must contain the logistics content, not merely 方案
    assert any("物流" in t and "報價" in t for t in top_texts), (
        f"Top results don't contain 物流+報價 context: {top_texts}"
    )


def test_investment_bond_bluechip_returns_keyword_message(searcher):
    """投資美國債券跟藍籌股 must return the investment message containing those keywords."""
    results = searcher.search("投資美國債券跟藍籌股", mode="hybrid", limit=5, expand=False, graph_expand=0, use_rerank=False)
    assert results, "No results returned"
    top_texts = [r.get("text", "") for r in results[:3]]
    assert any("債券" in t and "藍籌" in t for t in top_texts), (
        f"Top results don't contain keywords 債券+藍籌: {top_texts}"
    )


def test_bare_label_does_not_rank_all_label_docs(searcher):
    """Searching a bare label (product_inquiry) must not rank all product_inquiry docs."""
    results = searcher.search("product_inquiry", mode="hybrid", limit=10, expand=False, graph_expand=0, use_rerank=False)
    assert results, "No results returned"
    # Count how many top-10 results are product_inquiry
    labels = [r.get("metadata", {}).get("label") for r in results]
    pi_count = sum(1 for l in labels if l == "product_inquiry")
    # With the fix, label-only search should NOT rank every product_inquiry doc
    assert pi_count < len(results), (
        f"Bare label search ranked all {pi_count}/{len(results)} docs as product_inquiry"
    )


def test_label_filter_scopes_correctly(searcher):
    """Label filter investment_opportunity returns only investment_opportunity docs."""
    results = searcher.search("投資", mode="hybrid", limit=10, expand=False, graph_expand=0, use_rerank=False, label_filter="investment_opportunity")
    assert results, "No results returned"
    labels = [r.get("metadata", {}).get("label") for r in results]
    assert all(l == "investment_opportunity" for l in labels), (
        f"Label filter leaked other labels: {labels}"
    )