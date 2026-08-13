#!/usr/bin/env python3
"""Tests for the OA contract Streamlit app."""

import os
import sys
import types

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import apps.app as app_module


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Recorder:
    def __init__(self):
        self.titles = []
        self.subheaders = []
        self.dataframes = []
        self.errors = []

    def reset(self):
        self.titles.clear()
        self.subheaders.clear()
        self.dataframes.clear()
        self.errors.clear()


_recorder = _Recorder()


def _fake_streamlit():
    st = types.ModuleType("streamlit")
    st.set_page_config = lambda *a, **k: None
    st.title = lambda text, *a, **k: _recorder.titles.append(text)
    st.caption = lambda *a, **k: None
    st.subheader = lambda text, *a, **k: _recorder.subheaders.append(text)
    st.error = lambda text, *a, **k: _recorder.errors.append(text)
    st.stop = lambda: None
    st.sidebar = types.SimpleNamespace(
        text_input=lambda *a, **k: "/tmp/oa_index_t3",
        radio=lambda *a, **k: "Search",
    )
    st.text_input = lambda *a, **k: k.get("value", "")
    st.selectbox = lambda *a, **k: a[1][0] if len(a) > 1 and a[1] else None
    st.number_input = lambda *a, **k: k.get("value", 10)
    st.checkbox = lambda *a, **k: False
    st.multiselect = lambda *a, **k: []
    st.button = lambda *a, **k: True
    st.spinner = lambda *a, **k: _Ctx()
    st.success = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.dataframe = lambda df, *a, **k: _recorder.dataframes.append(df)
    st.expander = lambda *a, **k: _Ctx()
    st.json = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.metric = lambda *a, **k: None
    st.bar_chart = lambda *a, **k: None
    st.line_chart = lambda *a, **k: None
    st.columns = lambda n, *a, **k: [_Ctx() for _ in range(len(n) if isinstance(n, (list, tuple)) else n)]
    st.empty = lambda *a, **k: types.SimpleNamespace(markdown=lambda *a, **k: None, empty=lambda: None)
    st.session_state = {}
    return st


def test_agentic_contract_search_renders_results(monkeypatch):
    """Unified agentic UI: a contract question renders the evidence table."""
    _recorder.reset()
    fake_st = _fake_streamlit()
    monkeypatch.setattr(app_module, "st", fake_st)

    sample_results = [
        {
            "id": "1",
            "score": 0.9,
            "text": "Renewal agreement with Acme Corp",
            "metadata": {
                "counterparty_name": "Acme Corp",
                "department": "IT",
                "contract_type": "2",
                "requested_date": "2024-01-10",
                "status": "approved",
                "legal_approval": "yes",
                "overruled": "no",
                "title": "Renewal Contract",
                "ref_no": "OA-001",
            },
        }
    ]

    class _FakeSearcher:
        def __init__(self, embeddings):
            pass

        def search(self, **kwargs):
            return sample_results

    class _ContractAgent:
        def process(self, query, _stage=None):
            return {
                "output": "contract answer", "intent": "general",
                "tool": "contract_search",
                "tool_calls": [{"tool": "contract_search", "tool_input": query,
                                "filters": {}, "observation": "C"}],
                "steps": [], "success": True, "fallback": False,
                "clarify": False, "observation": "C",
            }

    monkeypatch.setattr(app_module, "_load_embeddings", lambda index_path: object())
    monkeypatch.setattr(app_module, "Searcher", _FakeSearcher)
    monkeypatch.setattr(app_module, "_build_agent", lambda path, emb: _ContractAgent())

    app_module._render_agentic("/tmp/oa_index_t3")

    assert _recorder.dataframes, "Agentic contract search should render an evidence table"
    df = _recorder.dataframes[-1]
    assert {"counterparty_name", "contract_type", "requested_date", "status"}.issubset(df.columns)
    assert not _recorder.errors


def test_no_chat_surface_in_ui():
    import inspect

    src = inspect.getsource(app_module)
    forbidden = ["Onyx Chat", "chat viewer", "chatbox", "iframe chat", "st.chat_message", "st.chat_input"]
    for token in forbidden:
        assert token not in src


def test_no_separate_search_tabs_regression():
    """Ticket 4 guard: the dual-tab Search / Risk Search sidebar views were
    replaced by one unified agentic workflow. Ensure they cannot come back."""
    import inspect

    src = inspect.getsource(app_module)
    # The removed dual-path render functions must stay removed.
    assert not hasattr(app_module, "_render_search")
    assert not hasattr(app_module, "_render_risk_search")
    # The removed sidebar tab labels must not reappear.
    for token in ['"Risk Search"', "_render_risk_search(", "_render_search("]:
        assert token not in src, "separate-tab behavior regressed: %s" % token
    # The unified agentic entrypoint must exist and be wired into main().
    assert hasattr(app_module, "_render_agentic")
    assert "_render_agentic(index_path)" in src



def test_exact_ref_number_shortcuts_to_single_contract(monkeypatch):
    import apps.app as app_module
    import json

    class FakeCursor:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, params=None): pass
        def fetchall(self): return self.conn.rows
        def close(self): pass

    class FakeConnection:
        def __init__(self):
            self.rows = [
                ("contract_1__chunk0", json.dumps({"ref_no": "CCA20250095", "contract_type": "type1"})),
                ("contract_2__chunk0", json.dumps({"ref_no": "CCA20250096", "contract_type": "type2"})),
            ]
        def cursor(self): return FakeCursor(self)

    class FakeEmb:
        database = type("DB", (), {"connection": FakeConnection()})()

    class FakeSearcher:
        embeddings = FakeEmb()
        def search(self, **kwargs): return []
        def _fetch_one_doc(self, doc_id):
            return {"id": doc_id, "metadata": {"ref_no": "CCA20250096", "contract_type": "type2"}}

    out = app_module._run_contract_search(FakeSearcher(), "CCA20250096", {})
    assert out is not None
    assert len(out) == 1
    assert out[0]["metadata"]["ref_no"] == "CCA20250096"



def test_contract_tool_exact_ref_match(monkeypatch):
    import apps.search_cli as cli
    import json

    class FakeCursor:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, params=None): pass
        def fetchall(self): return self.conn.rows
        def close(self): pass

    class FakeConnection:
        def __init__(self):
            self.rows = [
                ("contract_1__chunk0", json.dumps({"ref_no": "CCA20250095", "counterparty_name": "A"})),
                ("contract_2__chunk0", json.dumps({"ref_no": "CCA20250096", "counterparty_name": "B"})),
            ]
        def cursor(self): return FakeCursor(self)

    class FakeEmb:
        database = type("DB", (), {"connection": FakeConnection()})()

    class DummySearcher:
        embeddings = FakeEmb()
        def search(self, **kwargs):
            return [
                {"id": "contract_1__chunk0", "score": 0.2, "metadata": {"ref_no": "CCA20250095", "counterparty_name": "A"}, "text": "x"},
                {"id": "contract_2__chunk0", "score": 0.9, "metadata": {"ref_no": "CCA20250096", "counterparty_name": "B"}, "text": "y"},
            ]
        def _fetch_one_doc(self, doc_id):
            return {"id": doc_id, "metadata": {"ref_no": "CCA20250096", "counterparty_name": "B"}, "text": "y"}

    tool = cli.build_contract_tool(embeddings=object(), searcher=DummySearcher())
    out = tool("CCA20250096", {})
    assert "ref=CCA20250096" in out
    assert "CCA20250095" not in out


def test_agentic_contract_exact_ref_uses_last_query(monkeypatch):
    import apps.app as app_module

    calls = []
    class DummySearcher:
        def search(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setattr(app_module, "st", type("S", (), {"session_state": {"agentic_last_query": "CCA20250096"}, "spinner": lambda *a, **k: __import__("contextlib").nullcontext(), "info": lambda *a, **k: None, "warning": lambda *a, **k: None, "success": lambda *a, **k: None, "dataframe": lambda *a, **k: None, "expander": lambda *a, **k: __import__("contextlib").nullcontext(), "markdown": lambda *a, **k: None, "caption": lambda *a, **k: None, "columns": lambda *a, **k: [__import__("contextlib").nullcontext(), __import__("contextlib").nullcontext()], "json": lambda *a, **k: None, "toggle": lambda *a, **k: k.get("value"), "number_input": lambda *a, **k: k.get("value"), "multiselect": lambda *a, **k: k.get("default"), "selectbox": lambda *a, **k: None})())
    monkeypatch.setattr(app_module, "_run_contract_search", lambda searcher, query, filters: __import__("pandas").DataFrame([{"id":"contract_109__chunk0","ref_no":query,"metadata":{}}]))
    from apps.search.result_store import clear_results
    clear_results()  # force the legacy fallback path (no stash this turn)
    app_module._render_agentic_contract({"tool_calls": [{"tool_input": "contract_109__chunk0", "filters": {}}], "output": "ok", "success": True}, DummySearcher())
    assert calls == []



def test_service_exact_ref_no_bypass(monkeypatch):
    import json
    from apps.search.service import ContractSearchService

    meta = {"ref_no": "CCA20250096", "contract_type": "x"}

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn
            self.sql = ""
        def execute(self, sql, params=None):
            self.sql = sql
            self.params = params
        def fetchall(self):
            return self.conn.rows
        def fetchone(self):
            # _fetch_one_doc asks for (text, tags); exact-ref lookup asks for all rows.
            if self.sql and "text, tags" in self.sql.lower():
                return ("text", json.dumps(meta))
            return self.conn.rows[0] if self.conn.rows else None
        def close(self):
            pass

    class FakeConnection:
        def __init__(self):
            self.rows = [("contract_109__chunk0", "text", json.dumps(meta))]
        def cursor(self):
            return FakeCursor(self)

    class FakeDB:
        def __init__(self):
            self.connection = FakeConnection()

    class DummyEmb:
        graph = None
        database = FakeDB()
        def search(self, *a, **k):
            return []

    service = ContractSearchService(embeddings=DummyEmb())
    out = service.search("CCA20250096", {})
    assert len(out) == 1
    assert out[0]["metadata"]["ref_no"] == "CCA20250096"
