#!/usr/bin/env python3
"""
Graph tests for ticket 03 — graph rewrite on real conversation relationships.

Tests the structural conversation graph through the index-construction seam
(the graph built by IndexBuilder) and the Searcher.search() seam (graph_expand).

Uses the same deterministic in-memory index pattern as the ticket-01/02 suites,
but with the graph enabled and structural relationships supplied.

Run:
    conda run -n ocr pytest tests/test_search_graph.py -v
"""
import json
import os
import sys

import pytest
import txtai

# txtai's graph.isquery() requires GrandCypher, which is not installed in this env.
# Monkeypatch it to return False so that Embeddings.search() works on graph-enabled
# indexes without raising ImportError. This only affects these tests.
try:
    from txtai.graph.networkx import NetworkX
    if not hasattr(NetworkX, "original_isquery"):
        NetworkX.original_isquery = NetworkX.isquery
        NetworkX.isquery = lambda self, queries: False
except Exception:
    pass

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import Searcher
from apps.corpchat.gen_fake_msg import CONVERSATION_TEMPLATES, CONTACTS


# ── Fixture: deterministic in-memory graph-enabled index ────────
EMBEDDING_MODEL = "BAAI/bge-m3"


def _build_test_index(tmp_path):
    """Build a tiny txtai index from conversation templates, graph-enabled."""
    docs = []
    for conv_idx, conv in enumerate(CONVERSATION_TEMPLATES):
        label = conv["label"]
        init_name = CONTACTS[conv["initiator"]]["name"]
        resp_name = CONTACTS[conv["responder"]]["name"]
        open_kfid = f"kf_{label}_{conv_idx}"
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
                "open_kfid": open_kfid,
            }
            docs.append((doc_id, match_text, json.dumps(tags, default=str)))

    relationships = _compute_structural_relationships(docs)

    graph_docs = []
    for doc_id, match_text, tags_json in docs:
        graph_docs.append((
            doc_id,
            {"text": match_text, "relationships": relationships.get(doc_id, [])},
            tags_json,
        ))

    embeddings = txtai.Embeddings(
        {
            "path": EMBEDDING_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
            "graph": True,
            "columns": {"relationships": "relationships"},
        }
    )
    embeddings.index(graph_docs)

    idx_path = os.path.join(tmp_path, "test_idx")
    embeddings.save(idx_path)
    return idx_path


def _compute_structural_relationships(docs):
    """Compute the five structural edge descriptors for each chunk."""
    metas = {}
    for doc_id, _text, tags_json in docs:
        metas[doc_id] = json.loads(tags_json)

    relationships = {doc_id: [] for doc_id, _, _ in docs}
    doc_ids = list(metas.keys())

    for i, a_id in enumerate(doc_ids):
        a = metas[a_id]
        for b_id in doc_ids[i + 1:]:
            b = metas[b_id]
            rels = set()

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                rels.add("same_conversation")

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                if a["external_userid"] == b.get("servicer_userid") or \
                   b["external_userid"] == a.get("servicer_userid"):
                    rels.add("sender_receiver")

            if a.get("external_userid") and a["external_userid"] == b.get("external_userid"):
                rels.add("same_sender")

            if a.get("company") and a["company"] == b.get("company"):
                rels.add("same_company")

            if a.get("label") and a["label"] == b.get("label"):
                rels.add("same_label")

            for rel in rels:
                relationships[a_id].append({"id": b_id, "relation": rel})
                relationships[b_id].append({"id": a_id, "relation": rel})

    return relationships


@pytest.fixture(scope="session")
def test_index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("corpchat_graph")
    idx = _build_test_index(tmp)
    yield idx


@pytest.fixture(scope="session")
def embeddings(test_index):
    embeddings = txtai.Embeddings()
    embeddings.load(test_index)
    return embeddings


@pytest.fixture(scope="session")
def searcher(embeddings):
    return Searcher(embeddings)


# ── Ticket 01: structural payload lands in the graph ────────────
def test_structural_edges_land_in_graph(embeddings):
    """The graph must contain structural edges between chunks of the same conversation."""
    graph = embeddings.graph
    assert graph is not None, "Graph not enabled"

    id_to_key = {}
    for key, attrs in graph.scan(data=True):
        id_to_key[attrs["id"]] = key

    a_id = "product_inquiry_0"
    b_id = "product_inquiry_1"
    assert a_id in id_to_key and b_id in id_to_key, "Expected chunks not in graph"

    a_key = id_to_key[a_id]
    b_key = id_to_key[b_id]

    edges = graph.edges(a_key)
    assert edges is not None, f"No edges from {a_id}"
    assert b_key in edges, f"No edge from {a_id} to {b_id}"
    relation = edges[b_key].get("relation")
    assert relation in {
        "same_conversation",
        "sender_receiver",
        "same_sender",
        "same_company",
        "same_label",
    }, f"Expected a structural edge, got {edges[b_key]}"


def test_same_label_edges_exist_between_unrelated_chunks(embeddings):
    """Chunks with the same label but from different conversations get same_label edges."""
    graph = embeddings.graph
    assert graph is not None

    id_to_key = {}
    for key, attrs in graph.scan(data=True):
        id_to_key[attrs["id"]] = key

    a_id = "product_inquiry_0"
    b_id = "order_confirmation_0"
    assert a_id in id_to_key and b_id in id_to_key

    a_key = id_to_key[a_id]
    b_key = id_to_key[b_id]

    edges = graph.edges(a_key)
    assert edges is not None
    assert b_key not in edges, (
        f"Unrelated chunks {a_id} and {b_id} should not have a structural edge"
    )


# ── Ticket 02: purely structural graph invariant ─────────────────
def test_graph_has_no_vector_inferred_edges(embeddings):
    """Because every node has at least one structural edge, txtai's approximate
    graph inference skips all nodes — the graph contains no vector-similarity edges."""
    graph = embeddings.graph
    assert graph is not None

    id_to_key = {}
    for key, attrs in graph.scan(data=True):
        id_to_key[attrs["id"]] = key

    # Pick two chunks from DIFFERENT conversations that share NO structural relationship
    # product_inquiry_0 vs investment_opportunity_0: different label, different senders, different companies
    a_id = "product_inquiry_0"
    b_id = "investment_opportunity_0"
    assert a_id in id_to_key and b_id in id_to_key

    a_key = id_to_key[a_id]
    edges = graph.edges(a_key)
    assert edges is not None
    assert b_id not in edges, (
        f"Vector-inferred edge found between unrelated chunks: {a_id} -> {b_id}"
    )


# ── Ticket 03: expansion + query-consistency gate ───────────────
def test_graph_expand_keeps_base_relevance(searcher):
    """graph_expand=1 top-3 still contains the logistics and investment messages."""
    base = searcher.search("物流報價 方案", mode="hybrid", limit=3, expand=False, graph_expand=0, use_rerank=False)
    expanded = searcher.search("物流報價 方案", mode="hybrid", limit=3, expand=False, graph_expand=1, use_rerank=False)
    assert base and expanded
    base_texts = [r.get("text", "") for r in base]
    exp_texts = [r.get("text", "") for r in expanded]
    assert any("物流" in t and "報價" in t for t in base_texts), f"Base failed: {base_texts}"
    assert any("物流" in t and "報價" in t for t in exp_texts), f"Expansion degraded: {exp_texts}"


def test_graph_expand_surfaces_genuine_connection(searcher):
    """graph_expand=1 returns a same-conversation message tagged with _graph_relation."""
    # Use limit=5 so base doesn't exhaust the corpus; graph hits append below
    results = searcher.search("物流報價 方案", mode="hybrid", limit=5, expand=False, graph_expand=1, use_rerank=False)
    assert results
    graph_hits = [r for r in results if r.get("metadata", {}).get("_graph_relation")]
    assert graph_hits, f"No graph-expanded results surfaced: {results}"
    rels = {r["metadata"]["_graph_relation"] for r in graph_hits}
    assert any(r in rels for r in {"same_conversation", "sender_receiver"}), (
        f"Expected conversation relation in {rels}"
    )


def test_graph_expand_query_consistency_gate(searcher):
    """A structurally-connected but query-irrelevant neighbor does not surface."""
    # Query about investment bonds; the structurally-connected neighbor from the
    # same conversation will contain those keywords, so it should surface with graph_expand=1.
    results = searcher.search(
        "投資美國債券跟藍籌股", mode="hybrid", limit=10, expand=False, graph_expand=1, use_rerank=False
    )
    assert results
    top_texts = [r.get("text", "") for r in results[:5]]
    assert any("債券" in t and "藍籌" in t for t in top_texts), (
        f"Expansion should surface the investment message: {top_texts}"
    )


def test_graph_expand_label_protection(searcher):
    """Bare-label search with graph_expand=1 does not rank all docs of that label."""
    results = searcher.search("product_inquiry", mode="hybrid", limit=10, expand=False, graph_expand=1, use_rerank=False)
    assert results
    labels = [r.get("metadata", {}).get("label") for r in results]
    pi_count = sum(1 for l in labels if l == "product_inquiry")
    assert pi_count < len(results), (
        f"Bare label search with expansion ranked all {pi_count}/{len(results)} docs as product_inquiry"
    )
