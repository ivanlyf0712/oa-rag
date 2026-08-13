#!/usr/bin/env python3
"""
UI-flow tests for the Search page (`apps.corpchat.app._render_search_page`).

These tests catch four user-reported regressions:

  1. "Search was interrupted" is shown spuriously for a turn that is actively
     being processed (fresh `processing` turn rendered by `_render_chat_history`).
  2. Enhancement/Filters expanders auto-expand after a search completes
     (`expanded=not st.session_state.searching` pops them open).
  3. A greeting query still triggers a full search (no intent gate).
  4. The 6-stage progress window is shown for a greeting (same root cause as #3,
     but observable independently).

The seam: `_render_search_page()` is now a plain callable, so tests can drive it
with a recording fake `streamlit` module.

Run:
    conda run -n ocr pytest tests/test_search_ui.py -v
"""
import json
import os
import sys
import types

import pandas as pd
import pytest

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ── Recording fake streamlit ────────────────────────────────────────────────
class _Recorder:
    """Captures what the UI called, so tests can assert on rendered output."""

    def __init__(self):
        self.infos = []          # st.info(...) messages
        self.expander_expanded = []  # expanded= values passed to st.expander
        self.status_labels = []  # status.update(label=...) values
        self.search_calls = []   # _run_search invocations
        self.writes = []         # st.write(...) messages
        self.reruns = 0

    def reset(self):
        self.infos.clear()
        self.expander_expanded.clear()
        self.status_labels.clear()
        self.search_calls.clear()
        self.writes.clear()
        self.reruns = 0


def _make_fake_streamlit(recorder: _Recorder, search_impl=None):
    """Build a fake streamlit module backed by a _Recorder."""
    st = types.ModuleType("streamlit")

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    def _noop(*a, **k):
        return None

    def _noop_ctx(*a, **k):
        return _Ctx()

    class _FakeStatus(_Ctx):
        def __init__(self):
            super().__init__()
        def update(self, **kwargs):
            recorder.status_labels.append(kwargs.get("label"))

    # Page setup
    st.set_page_config = _noop
    st.title = _noop
    st.markdown = _noop
    st.caption = _noop
    st.subheader = _noop
    st.info = lambda *a, **k: recorder.infos.append(a[0]) if a else None
    st.warning = _noop
    st.error = _noop
    st.success = _noop
    st.divider = _noop
    st.write = lambda *a, **k: recorder.writes.extend(str(x) for x in a if x)

    # Layout context managers
    st.expander = lambda *a, **k: (_Ctx(), recorder.expander_expanded.append(k.get("expanded")))[0]
    st.chat_message = _noop_ctx
    st.status = lambda *a, **k: _FakeStatus()
    st.tabs = lambda *a, **k: [_Ctx() for _ in (a[0] if a and isinstance(a[0], (list, tuple)) else a)]
    st.columns = lambda *a, **k: [_Ctx() for _ in range(len(a[0]) if a and isinstance(a[0], (list, tuple)) else (a[0] if a else 1))]

    # Widgets
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: True
    st.radio = lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None)
    st.selectbox = lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None)
    st.text_input = lambda *a, **k: ""
    st.chat_input = lambda *a, **k: None
    st.number_input = lambda *a, **k: 1
    st.slider = lambda *a, **k: 10
    st.dataframe = _noop
    st.metric = _noop
    st.bar_chart = _noop
    st.iframe = _noop
    st.rerun = lambda: setattr(recorder, "reruns", recorder.reruns + 1)

    # Session state
    class _FS(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value
    st.session_state = _FS()

    st.cache_data = lambda *a, **k: (a[0] if a else (lambda f: f))
    st.cache_resource = lambda *a, **k: (a[0] if a else (lambda f: f))

    # Sidebar
    class _FakeSidebar:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        title = caption = divider = _noop
        radio = lambda *a, **k: "Search"
        checkbox = lambda *a, **k: True
        slider = lambda *a, **k: 10
        text_input = lambda *a, **k: ""
        def expander(self, *a, **k): return _Ctx()
    st.sidebar = _FakeSidebar()

    return st


# ── Install fake streamlit + import app ─────────────────────────────────────
_recorder = _Recorder()
sys.modules["streamlit"] = _make_fake_streamlit(_recorder)

# Mock DB so app.py imports cleanly
import core.db as core_db_module
class _FakeCursor:
    def execute(self, sql, *args): return None
    def fetchone(self): return (0, 0, 0, [])
    def fetchall(self): return []
    def close(self): return None
class _FakeConn:
    def cursor(self): return _FakeCursor()
    def close(self): return None
core_db_module.get_db_connection = lambda: _FakeConn()
pd.read_sql = lambda *a, **k: pd.DataFrame()

from apps.corpchat import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def _bind_recording_st(monkeypatch):
    """Bind the app module's `st` to the live recording fake for every test.

    The app module does `import streamlit as st` at import time, capturing a
    reference to whatever fake was in sys.modules then. When test_app_search.py
    and this file both install fakes, app_module.st can point at a stale fake.
    Rebinding per-test makes the page render through the recording fake, so the
    session_state the test populates is the one the page actually reads.
    """
    monkeypatch.setattr(app_module, "st", sys.modules["streamlit"])
    yield
    monkeypatch.undo()


# ── Fixture: session state with a fresh processing turn ─────────────────────
def _fresh_session(query: str):
    """Return a session_state with one fresh processing turn (as if just submitted)."""
    ss = type("SS", (dict,), {})()
    ss["chat_history"] = [{"query": query, "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True
    return ss


@pytest.fixture(autouse=True)
def _clean_recorder():
    _recorder.reset()
    yield
    _recorder.reset()


# ── Bug 1: spurious "Search was interrupted" ─────────────────────────────────
def test_no_spurious_interrupted_for_fresh_processing_turn(monkeypatch):
    """A freshly-submitted processing turn must NOT render 'Search was interrupted'."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{"query": "Hi", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True

    # Stub out the search/LLM so the pending-turn handler is fast & deterministic
    monkeypatch.setattr(app_module, "_run_search", lambda *a, **k: ([], []))
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "fallback")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    # The write() in the status block will call write(); ensure Searcher isn't
    # reached by _run_search (already stubbed). Render the page.
    app_module._render_search_page()

    assert not _recorder.infos, (
        f"'Search was interrupted' (or other st.info) rendered spuriously: {_recorder.infos}"
    )


# ── Bug 2: filters/expanders must not auto-expand after search ───────────────
def test_expanders_not_expanded_when_idle(monkeypatch):
    """When not searching, Enhancement/Filters expanders must be collapsed."""
    from streamlit import session_state as ss
    ss["chat_history"] = []
    ss["searching"] = False
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    assert _recorder.expander_expanded, "No expander calls recorded"
    assert all(e is False for e in _recorder.expander_expanded), (
        f"Expanders auto-expanded when idle: {_recorder.expander_expanded}"
    )


# ── Bug 3: greeting must not trigger search ──────────────────────────────────
def test_greeting_does_not_call_search(monkeypatch):
    """A greeting query must NOT invoke _run_search (no wasted search)."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{"query": "Hi", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True

    # Spy on _run_search — if the greeting path is correct, it is never called.
    calls = []
    def _spy_run_search(*a, **k):
        calls.append(a[0])
        return ([], [])
    monkeypatch.setattr(app_module, "_run_search", _spy_run_search)
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    assert calls == [], f"_run_search called for greeting: {calls}"


def test_greeting_does_not_show_progress_window(monkeypatch):
    """A greeting must not render the 6-stage progress status window."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{"query": "Hi", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True

    monkeypatch.setattr(app_module, "_run_search", lambda *a, **k: ([], []))
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    # No status window should have appeared for a greeting
    assert _recorder.status_labels == [], (
        f"Progress status shown for greeting: {_recorder.status_labels}"
    )


# ── Search intent still works (no regression) ───────────────────────────────
def test_search_query_still_calls_search(monkeypatch):
    """A real search query (with explicit search keyword) must still invoke _run_search."""
    from streamlit import session_state as ss
    # "找" is an explicit search keyword → rule-classified as search, deterministic
    ss["chat_history"] = [{"query": "找物流報價 方案", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True

    calls = []
    def _spy_run_search(*a, **k):
        calls.append(a[0])
        return ([], [])
    monkeypatch.setattr(app_module, "_run_search", _spy_run_search)
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    assert calls == ["找物流報價 方案"], f"Search query did not reach _run_search: {calls}"
