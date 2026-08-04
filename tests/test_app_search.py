#!/usr/bin/env python3
"""
App-layer tests for tickets 01–04 — verify the four enhancement layers are
wired through app.py's public search entry point `search_messages_onyx()`.

Seam: `apps.corpchat.app.search_messages_onyx()` — the function the Streamlit
UI calls. We mock `streamlit` (so app.py can be imported without a running
server), mock `_load_search_index` (so no real index is needed), and use
deterministic fakes for QueryExpander / Reranker so no live API or model
download is required.

The four tickets verified:
  - Ticket 01: Chinese-capable hybrid base (物流報價 方案 → product_inquiry)
  - Ticket 02: LLM query expansion (expand=True path)
  - Ticket 03: graph expansion (graph_expand=1 path)
  - Ticket 04: multilingual reranker (use_rerank=True path)

Run:
    conda run -n ocr pytest tests/test_app_search.py -v
"""
import json
import os
import sys
import types

import pandas as pd
import pytest
import txtai

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.gen_fake_msg import CONVERSATION_TEMPLATES, CONTACTS


# ── Mock DB access so app.py's module-level UI code imports cleanly ──
class _FakeCursor:
    def execute(self, sql, *args):
        return None
    def fetchone(self):
        return (0, 0, 0, [])
    def fetchall(self):
        return []
    def close(self):
        return None


class _FakeConn:
    def cursor(self):
        return _FakeCursor()
    def close(self):
        return None


import core.db as core_db_module
core_db_module.get_db_connection = lambda: _FakeConn()

# Replace pandas.read_sql globally so fetch_contacts/fetch_messages return empty frames
def _fake_read_sql(*args, **kwargs):
    return pd.DataFrame()
pd.read_sql = _fake_read_sql


# ── Mock streamlit so app.py can be imported without a server ────
class _FakeSessionState(dict):
    """A dict that also supports attribute access + in checks, like st.session_state."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def _make_fake_streamlit():
    """Build a minimal fake streamlit module for import-time use."""
    st = types.ModuleType("streamlit")

    def _noop(*args, **kwargs):
        return None

    def _noop_context(*args, **kwargs):
        class _Ctx:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        return _Ctx()

    st.set_page_config = _noop
    st.title = _noop
    st.markdown = _noop
    st.caption = _noop
    st.subheader = _noop
    st.info = _noop
    st.warning = _noop
    st.error = _noop
    st.success = _noop
    st.divider = _noop
    st.spinner = _noop_context
    st.expander = _noop_context
    st.chat_message = _noop_context
    st.tabs = lambda *a, **k: [_noop_context() for _ in (a[0] if a and isinstance(a[0], (list, tuple)) else a)]
    st.columns = lambda *a, **k: [_noop_context() for _ in range(len(a[0]) if a and isinstance(a[0], (list, tuple)) else (a[0] if a else 1))]
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: True
    st.selectbox = lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None)
    st.text_input = lambda *a, **k: ""
    st.number_input = lambda *a, **k: 1
    st.slider = lambda *a, **k: 10
    st.dataframe = _noop
    st.metric = _noop
    st.bar_chart = _noop
    st.iframe = _noop
    st.session_state = _FakeSessionState(
        search_results=None, search_query=None, rag_answer=None, selected_kfid=None
    )
    st.cache_data = lambda *a, **k: (a[0] if a else (lambda f: f))
    st.cache_resource = lambda *a, **k: (a[0] if a else (lambda f: f))
    return st


# Install the fake streamlit BEFORE importing app.py
sys.modules["streamlit"] = _make_fake_streamlit()

from apps.corpchat import app as app_module  # noqa: E402


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
    tmp = tmp_path_factory.mktemp("corpchat_app")
    idx = _build_test_index(tmp)
    yield idx


@pytest.fixture(scope="session")
def embeddings(test_index):
    embeddings = txtai.Embeddings()
    embeddings.load(test_index)
    return embeddings


@pytest.fixture()
def app_searcher(monkeypatch, embeddings):
    """Wire app.search_messages_onyx to a deterministic in-memory index."""
    # Replace _load_search_index so no real index is needed
    monkeypatch.setattr(app_module, "_load_search_index", lambda: embeddings)
    return app_module


# ── Ticket 01: base retrieval through app.py ────────────────────
def test_app_base_logistics_quotation(monkeypatch, app_searcher, embeddings):
    """app.py search must surface the logistics message (base works)."""
    # Disable all enhancement layers to test the base path
    results = app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=False, expand=False, graph_expand=0
    )
    assert results, "No results returned"
    top_texts = [r[1] for r in results[:3]]
    assert any("物流" in t and "報價" in t for t in top_texts), (
        f"Base failed through app.py: {top_texts}"
    )


def test_app_base_investment_bond(monkeypatch, app_searcher, embeddings):
    """app.py search must surface the investment message (base works)."""
    results = app_searcher.search_messages_onyx(
        "投資美國債券跟藍籌股", top_k=5, use_rerank=False, expand=False, graph_expand=0
    )
    assert results, "No results returned"
    top_texts = [r[1] for r in results[:3]]
    assert any("債券" in t and "藍籌" in t for t in top_texts), (
        f"Base failed through app.py: {top_texts}"
    )


# ── Ticket 02: LLM expansion wired through app.py ───────────────
def test_app_expansion_wired(monkeypatch, app_searcher, embeddings):
    """app.py must construct a QueryExpander when expand=True."""
    import apps.corpchat.search as search_module
    from apps.corpchat.search import QueryExpander

    captured = {}

    def _fake_expander(*args, **kwargs):
        captured["constructed"] = True
        return QueryExpander(*args, **kwargs)

    monkeypatch.setattr(search_module, "QueryExpander", _fake_expander)

    # expand=True must construct the expander
    app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=False, expand=True, graph_expand=0
    )
    assert captured.get("constructed"), "QueryExpander was not constructed when expand=True"


def test_app_expansion_improves_or_matches_base(monkeypatch, app_searcher, embeddings):
    """app.py with expand=True must be at least as relevant as expand=False."""
    query = "物流報價 方案"
    base = app_searcher.search_messages_onyx(
        query, top_k=5, use_rerank=False, expand=False, graph_expand=0
    )
    expanded = app_searcher.search_messages_onyx(
        query, top_k=5, use_rerank=False, expand=True, graph_expand=0
    )
    assert base and expanded, "No results returned"

    base_top = [r[1] for r in base[:3]]
    exp_top = [r[1] for r in expanded[:3]]
    assert any("物流" in t and "報價" in t for t in base_top), f"Base failed: {base_top}"
    assert any("物流" in t and "報價" in t for t in exp_top), f"Expansion degraded: {exp_top}"


# ── Ticket 03: graph expansion wired through app.py ─────────────
def test_app_graph_expand_wired(monkeypatch, app_searcher, embeddings):
    """app.py must pass graph_expand through to Searcher.search()."""
    from apps.corpchat.search import Searcher

    captured = {}

    original_search = Searcher.search

    def _spy_search(self, *args, **kwargs):
        captured["graph_expand"] = kwargs.get("graph_expand")
        return original_search(self, *args, **kwargs)

    monkeypatch.setattr(Searcher, "search", _spy_search)

    app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=False, expand=False, graph_expand=1
    )
    assert captured.get("graph_expand") == 1, (
        f"graph_expand not passed through: {captured.get('graph_expand')}"
    )


# ── Ticket 04: reranker wired through app.py ────────────────────
def test_app_rerank_wired(monkeypatch, app_searcher, embeddings):
    """app.py must construct a Reranker when use_rerank=True."""
    import apps.corpchat.search as search_module
    from apps.corpchat.search import Reranker

    captured = {}

    def _fake_reranker(*args, **kwargs):
        captured["constructed"] = True
        return Reranker(*args, **kwargs)

    monkeypatch.setattr(search_module, "Reranker", _fake_reranker)

    app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=True, expand=False, graph_expand=0
    )
    assert captured.get("constructed"), "Reranker was not constructed when use_rerank=True"


def test_app_rerank_improves_or_matches_base(monkeypatch, app_searcher, embeddings):
    """app.py with use_rerank=True must be at least as relevant as use_rerank=False."""
    query = "投資美國債券跟藍籌股"
    base = app_searcher.search_messages_onyx(
        query, top_k=5, use_rerank=False, expand=False, graph_expand=0
    )
    reranked = app_searcher.search_messages_onyx(
        query, top_k=5, use_rerank=True, expand=False, graph_expand=0
    )
    assert base and reranked, "No results returned"

    base_top = [r[1] for r in base[:3]]
    reranked_top = [r[1] for r in reranked[:3]]
    assert any("債券" in t and "藍籌" in t for t in base_top), f"Base failed: {base_top}"
    assert any("債券" in t and "藍籌" in t for t in reranked_top), (
        f"Rerank degraded: {reranked_top}"
    )


# ── Graceful DB failure (app must not crash when Postgres is down) ──
def test_app_db_down_does_not_crash(monkeypatch):
    """Helpers must return empty data instead of raising when the DB is down."""
    def _raise(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(app_module, "get_db_connection", _raise)

    # Each helper must degrade gracefully
    df_contacts = app_module.fetch_contacts()
    assert df_contacts.empty, "fetch_contacts should return empty DataFrame on DB failure"

    df_msgs = app_module.fetch_messages()
    assert df_msgs.empty, "fetch_messages should return empty DataFrame on DB failure"

    stats = app_module.fetch_stats()
    assert stats == (0, 0, 0, []), f"fetch_stats should return zeros on DB failure: {stats}"

    name_map = app_module.get_contact_name_map()
    assert name_map == {}, "get_contact_name_map should return {} on DB failure"

    convs = app_module.get_conversation_list()
    assert convs == [], "get_conversation_list should return [] on DB failure"

    msgs = app_module.get_messages_for_conversation("kf_test_0")
    assert msgs.empty, "get_messages_for_conversation should return empty DataFrame on DB failure"


# ── Agentic wiring through app.py ───────────────────────────────
def test_app_agentic_wired(monkeypatch, app_searcher, embeddings):
    """app.py must construct an AgenticDecider when agentic=True."""
    import apps.corpchat.search as search_module
    from apps.corpchat.search import AgenticDecider

    captured = {}

    def _fake_decider(*args, **kwargs):
        captured["constructed"] = True
        return AgenticDecider(*args, **kwargs)

    monkeypatch.setattr(search_module, "AgenticDecider", _fake_decider)

    app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=False, expand=False, graph_expand=0,
        agentic=True
    )
    assert captured.get("constructed"), "AgenticDecider was not constructed when agentic=True"


def test_app_agentic_decision_overrides_manual_params(monkeypatch, app_searcher, embeddings):
    """agentic=True must let AgenticDecider override mode/expand/graph/rerank."""
    import apps.corpchat.search as search_module
    from apps.corpchat.search import Searcher

    captured = {}

    class _FakeDecider:
        def decide(self, query):
            return {"mode": "keyword", "expand": False, "graph_expand": 0, "use_rerank": False}

    monkeypatch.setattr(search_module, "AgenticDecider", _FakeDecider)

    original_search = Searcher.search

    def _spy_search(self, *args, **kwargs):
        captured["mode"] = kwargs.get("mode")
        captured["expand"] = kwargs.get("expand")
        captured["graph_expand"] = kwargs.get("graph_expand")
        captured["use_rerank"] = kwargs.get("use_rerank")
        return original_search(self, *args, **kwargs)

    monkeypatch.setattr(Searcher, "search", _spy_search)

    # Pass manual params that should be overridden by the agentic decision
    app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=True, expand=True, graph_expand=1,
        agentic=True
    )
    assert captured.get("mode") == "keyword", f"mode not overridden: {captured.get('mode')}"
    assert captured.get("expand") is False, f"expand not overridden: {captured.get('expand')}"
    assert captured.get("graph_expand") == 0, f"graph_expand not overridden: {captured.get('graph_expand')}"
    assert captured.get("use_rerank") is False, f"use_rerank not overridden: {captured.get('use_rerank')}"


def test_app_agentic_defaults_to_false(monkeypatch, app_searcher, embeddings):
    """agentic must default to False — manual params are used unless toggled."""
    import apps.corpchat.search as search_module
    from apps.corpchat.search import Searcher

    captured = {}

    class _FakeDecider:
        def decide(self, query):
            captured["called"] = True
            return {"mode": "keyword", "expand": False, "graph_expand": 0, "use_rerank": False}

    monkeypatch.setattr(search_module, "AgenticDecider", _FakeDecider)

    original_search = Searcher.search

    def _spy_search(self, *args, **kwargs):
        captured["mode"] = kwargs.get("mode")
        return original_search(self, *args, **kwargs)

    monkeypatch.setattr(Searcher, "search", _spy_search)

    # No agentic param → default False → manual hybrid mode, decider NOT called
    app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=False, expand=False, graph_expand=0
    )
    assert "called" not in captured, "AgenticDecider should not be called when agentic=False"
    assert captured.get("mode") == "hybrid", f"mode should be hybrid by default: {captured.get('mode')}"


# ── Full pipeline: all four tickets together ────────────────────
def test_app_full_pipeline_all_layers(monkeypatch, app_searcher, embeddings):
    """app.py default (expand=True, use_rerank=True, graph_expand=1) must work."""
    results = app_searcher.search_messages_onyx(
        "物流報價 方案", top_k=5, use_rerank=True, expand=True, graph_expand=1
    )
    assert results, "No results returned"
    top_texts = [r[1] for r in results[:3]]
    assert any("物流" in t and "報價" in t for t in top_texts), (
        f"Full pipeline lost the logistics message: {top_texts}"
    )


def test_app_label_filter_still_scopes(monkeypatch, app_searcher, embeddings):
    """Label filter must still scope correctly through app.py."""
    results = app_searcher.search_messages_onyx(
        "投資", top_k=10, use_rerank=False, expand=False, graph_expand=0,
        label_filter="investment_opportunity"
    )
    assert results, "No results returned"
    labels = [r[5] for r in results]
    assert all(l == "investment_opportunity" for l in labels), (
        f"Label filter leaked other labels: {labels}"
    )