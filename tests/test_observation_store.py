"""Tests for ticket 02: observation formatter + result store + agent wiring.

Run:
    venv/bin/python -m pytest tests/test_observation_store.py -v
"""
import json
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.result_store import clear_results, snapshot_results, stash_results
from apps.search.service import (
    OBSERVATION_ROW_BUDGET,
    format_contract_observation,
)


@pytest.fixture(autouse=True)
def _clean_store():
    clear_results()
    yield
    clear_results()


def _row(ref, **meta):
    m = {"ref_no": ref, "risk_score": 0, "risk_severity": "low"}
    m.update(meta)
    return {"id": ref, "text": meta.pop("text", "snippet text"), "metadata": m}


# -- observation formatter -------------------------------------------------

def test_observation_carries_rich_fields():
    rows = [_row("CCA00000001", counterparty_name="Alpha Corp", contract_type="2",
                 status_label="completed", contract_start_date="2024-01-01",
                 contract_end_date="2025-06-30", amount_label="HK$6,000,000",
                 risk_score=75, risk_severity="high",
                 matched_signals=["IsRisksAccepted = no (+50)", "Over5M = yes (+25)"])]
    out = format_contract_observation(rows)
    assert "ref=CCA00000001" in out
    assert "Alpha Corp" in out
    assert "type=2" in out
    assert "status=completed" in out
    assert "2024-01-01 -> 2025-06-30" in out
    assert "amount=HK$6,000,000" in out
    assert "risk=75 (high)" in out
    assert "IsRisksAccepted = no (+50)" in out
    assert "snippet text" in out


def test_observation_empty_is_empty_string():
    assert format_contract_observation([]) == ""


def test_observation_budget_and_overflow_marker():
    rows = [_row("R%03d" % i) for i in range(OBSERVATION_ROW_BUDGET + 12)]
    out = format_contract_observation(rows)
    lines = out.splitlines()
    assert len(lines) == OBSERVATION_ROW_BUDGET + 1
    assert lines[-1] == "(+12 more contracts not shown - see results table)"


def test_observation_no_overflow_marker_within_budget():
    rows = [_row("R1"), _row("R2")]
    out = format_contract_observation(rows)
    assert "more contracts not shown" not in out
    assert len(out.splitlines()) == 2


# -- result store ----------------------------------------------------------

def test_store_stash_snapshot_clear():
    stash_results([{"id": "a"}], query="q1", filters={"status": "active"},
                  rank_by="risk", observation_count=1)
    snap = snapshot_results()
    assert snap["total"] == 1
    assert snap["query"] == "q1"
    assert snap["filters"] == {"status": "active"}
    assert snap["rank_by"] == "risk"
    clear_results()
    snap = snapshot_results()
    assert snap["total"] == 0 and snap["rows"] == []


def test_store_snapshot_is_a_copy():
    rows = [{"id": "a"}]
    stash_results(rows, query="q")
    snap = snapshot_results()
    snap["rows"].append({"id": "b"})
    rows.append({"id": "c"})
    assert snapshot_results()["total"] == 1


def test_store_latest_stash_wins():
    stash_results([{"id": "a"}], query="first")
    stash_results([{"id": "b"}, {"id": "c"}], query="second")
    snap = snapshot_results()
    assert snap["query"] == "second" and snap["total"] == 2


# -- unified tool adapter ---------------------------------------------------

class _FakeSearcher:
    def __init__(self, results):
        self.embeddings = None
        self._results = results
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._results)

    def _fetch_one_doc(self, doc_id):
        return None


def _semantic_rows():
    return [
        {"id": "a", "text": "liability clause", "score": 0.9,
         "metadata": {"ref_no": "R1", "counterparty_name": "Alpha",
                      "decoded_fields": {"IsRisksAccepted": {"label": "No"}}}},
        {"id": "b", "text": "renewal clause", "score": 0.8,
         "metadata": {"ref_no": "R2", "counterparty_name": "Beta"}},
    ]


def test_contract_tool_stashes_full_rows_and_returns_observation():
    from apps.search_cli import build_contract_tool

    searcher = _FakeSearcher(_semantic_rows())
    tool = build_contract_tool(embeddings=None, searcher=searcher)
    out = tool("renewal terms", {})  # no risk keywords -> semantic path

    # observation is the formatted, budget-capped text
    assert "ref=R1" in out and "ref=R2" in out
    assert "risk=" in out  # risk fields present for every row
    # the store carries the full structured rows for the UI
    snap = snapshot_results()
    assert snap["total"] == 2
    assert snap["query"] == "renewal terms"
    assert snap["rows"][0]["metadata"]["ref_no"] == "R1"
    assert snap["rows"][0]["risk_score"] > 0  # unconditional scoring


def test_risk_tool_shim_delegates_to_unified_tool():
    from apps.search_cli import build_risk_tool

    # shim with neither embeddings nor searcher -> the underlying service
    # raises a clear error at build time rather than silently succeeding
    with pytest.raises(ValueError):
        build_risk_tool(embeddings=None)


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.connection = self

    def cursor(self):
        return self

    def execute(self, sql):
        if "text" in sql:  # emulate a 2-column sections table
            raise Exception("no such column: text")
        return self

    def fetchall(self):
        return self._rows

    def close(self):
        pass


def test_risk_language_flows_through_unified_tool_no_mode_gate():
    from apps.search_cli import build_contract_tool

    db_rows = [
        ("s1", json.dumps({"contract_id": "A", "ref_no": "R1",
                           "counterparty_name": "Alpha",
                           "decoded_fields": {"IsRisksAccepted": {"label": "No"}}})),
        ("s2", json.dumps({"contract_id": "B", "ref_no": "R2",
                           "counterparty_name": "Beta",
                           "decoded_fields": {"IsRisksAccepted": {"label": "Yes"}}})),
    ]

    class FakeEmb:
        database = _FakeDB(db_rows)

    searcher = _FakeSearcher([])
    searcher.embeddings = FakeEmb()
    tool = build_contract_tool(embeddings=None, searcher=searcher)

    out = tool("contracts where risk was not accepted", {})

    assert "ref=R1" in out          # the flagged contract is returned
    assert "ref=R2" not in out     # risk filter applied, not gated to empty
    snap = snapshot_results()
    assert snap["rank_by"] == "risk"  # planner rank hint surfaced
    assert snap["total"] == 1


# -- agent synthesis: no 5-line truncation ----------------------------------

class _CaptureLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(messages)
        return type("R", (), {"content": "a concise summary"})()


def test_default_synthesize_sends_full_observation():
    from apps.search.langchain_agent import LangChainAgent

    obs = chr(10).join(
        "%d. [ref=R%d | Party%d] risk=%d (low); signals: none; snippet" % (i, i, i, i)
        for i in range(1, 13)  # 12 evidence lines, well over the old 5-line cap
    )
    llm = _CaptureLLM()
    agent = LangChainAgent(contract_tool=lambda q, f: obs, llm=llm)
    out = agent._default_synthesize("query", "contract_search", obs)

    assert out == "a concise summary"
    prompt = llm.prompts[0][0][1]
    assert "[ref=R12 | Party12]" in prompt  # line 12 survived -> no truncation


def test_default_synthesize_empty_observation():
    from apps.search.langchain_agent import LangChainAgent

    agent = LangChainAgent(contract_tool=lambda q, f: "", llm=_CaptureLLM())
    assert agent._default_synthesize("q", "contract_search", "") == \
        "No matching contracts were found."


def test_unified_system_prompt_when_no_risk_tool():
    from apps.search.langchain_agent import LangChainAgent

    agent = LangChainAgent(contract_tool=lambda q, f: "", llm=_CaptureLLM())
    assert "risk filters and risk ranking are extracted automatically" in \
        agent._decision_system()
    agent_with_risk = LangChainAgent(contract_tool=lambda q, f: "",
                                     risk_tool=lambda q: "", llm=_CaptureLLM())
    assert "call risk_search" in agent_with_risk._decision_system()


# -- app wiring -------------------------------------------------------------

def test_app_risk_tool_no_longer_wired():
    import inspect
    import apps.app as app_module

    src = inspect.getsource(app_module)
    # the regex observation parser is deleted
    assert not hasattr(app_module, "_render_risk_evidence")
    assert "risk_score=(\\S+)" not in src
    # the risk tool is not constructed anymore
    assert "build_risk_tool(" not in src
    assert "_render_risk_evidence(" not in src


def test_render_agentic_contract_uses_result_store(monkeypatch):
    import contextlib
    import pandas as pd
    import apps.app as app_module

    captured = {}

    class FakeSt:
        session_state = {"agentic_last_query": "liability"}

        @staticmethod
        def spinner(*a, **k): return contextlib.nullcontext()

        @staticmethod
        def expander(*a, **k): return contextlib.nullcontext()

        @staticmethod
        def info(*a, **k): pass

        @staticmethod
        def warning(*a, **k): pass

        @staticmethod
        def success(msg): captured["success"] = msg

        @staticmethod
        def markdown(*a, **k): pass

        @staticmethod
        def caption(*a, **k): pass

        @staticmethod
        def dataframe(df, **k): captured["df"] = df

        @staticmethod
        def selectbox(*a, **k): return None

        @staticmethod
        def columns(spec): return [contextlib.nullcontext() for _ in (spec if isinstance(spec, list) else range(spec))]

        @staticmethod
        def toggle(label, value=False, key=None):
            captured.setdefault("toggle_default", value)
            return True  # enable rank_by_risk so number_input appears

        @staticmethod
        def number_input(label, min_value=None, max_value=None, value=None, step=None, key=None):
            captured.setdefault("min_score_default", value)
            return value

        @staticmethod
        def multiselect(label, options, default=None, key=None):
            captured.setdefault("columns_default", default)
            return default

    class DummySearcher:
        def search(self, **kwargs):
            raise AssertionError("UI must render from the store, not re-query")

    stash_results(
        [{"id": "a", "text": "t", "score": 1.0,
          "metadata": {"ref_no": "R1", "counterparty_name": "Alpha",
                       "risk_score": 95, "risk_severity": "high"}}],
        query="liability",
    )
    monkeypatch.setattr(app_module, "st", FakeSt)
    app_module._render_agentic_contract(
        {"tool_calls": [{"tool": "contract_search", "tool_input": "liability",
                         "filters": {}}],
         "output": "summary", "success": True},
        DummySearcher(),
    )
    assert captured["success"] == "1 of 1 supporting contract(s) shown"
    df = captured["df"]
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["ref_no"] == "R1"
    # min_score is stored in session_state; number_input called with DEFAULT_MIN_RISK_SCORE
    assert captured["min_score_default"] == 80
    # Column picker was moved to sidebar (tested in main() scope, not here)


def test_render_agentic_contract_falls_back_without_stash(monkeypatch):
    import contextlib
    import pandas as pd
    import apps.app as app_module

    class FakeSt:
        session_state = {"agentic_last_query": "q"}

        @staticmethod
        def spinner(*a, **k): return contextlib.nullcontext()

        @staticmethod
        def expander(*a, **k): return contextlib.nullcontext()

        @staticmethod
        def info(*a, **k): pass

        @staticmethod
        def warning(*a, **k): pass

        @staticmethod
        def success(*a, **k): pass

        @staticmethod
        def markdown(*a, **k): pass

        @staticmethod
        def caption(*a, **k): pass

        @staticmethod
        def dataframe(*a, **k): pass

        @staticmethod
        def selectbox(*a, **k): return None

    class DummySearcher:
        def search(self, **kwargs):
            return []

    called = {}
    monkeypatch.setattr(app_module, "st", FakeSt)
    monkeypatch.setattr(
        app_module, "_run_contract_search",
        lambda searcher, query, filters: (
            called.setdefault("hit", True),
            pd.DataFrame([{"id": "x", "ref_no": "R9", "metadata": {}}]),
        )[1],
    )
    clear_results()
    app_module._render_agentic_contract(
        {"tool_calls": [{"tool": "contract_search", "tool_input": "q", "filters": {}}],
         "output": "ok", "success": True},
        DummySearcher(),
    )
    assert called.get("hit") is True  # legacy fallback path preserved
