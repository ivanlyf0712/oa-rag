#!/usr/bin/env python3
"""Ticket 8 — tests for the risky-contract search pipeline."""

import os
import sys
import types

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.risk_search import (
    DEFAULT_CLARIFICATION,
    MODE_CLARIFY,
    MODE_GENERAL,
    MODE_RISKY,
    RiskPlanner,
    apply_risk_filters,
    gate_and_sort,
    normalize_filter_value,
    run_risk_search,
    score_risk,
    validate_filters,
)


class _FakeSessionState(dict):
    """Dict with attribute access + `in` support, like st.session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# ═══════════════════════════════════════════════════════════════════
# Ticket 1 — planner contract + allowlist validation
# ═══════════════════════════════════════════════════════════════════

def test_allowlist_accepts_known_fields():
    filters = [{"field": "IsRisksAccepted", "value": "no"},
               {"field": "FlagNeedLegal", "value": "YES"},
               {"field": "Over5M", "value": "na"}]
    clean = validate_filters(filters)
    assert [c["field"] for c in clean] == ["IsRisksAccepted", "FlagNeedLegal", "Over5M"]
    assert [c["value"] for c in clean] == ["no", "yes", "na"]  # normalized


def test_allowlist_rejects_unknown_fields_and_values():
    filters = [{"field": "HackField", "value": "yes"},
               {"field": "Over5M", "value": "maybe"},
               {"field": "IsRisksAccepted", "value": "no"}]
    clean = validate_filters(filters)
    assert clean == [{"field": "IsRisksAccepted", "op": "=", "value": "no"}]


def test_normalize_filter_value():
    assert normalize_filter_value("YES") == "yes"
    assert normalize_filter_value("Not Accepted") == "no"
    assert normalize_filter_value("N/A") == "na"
    assert normalize_filter_value("bogus") is None
    assert normalize_filter_value(None) is None


# ═══════════════════════════════════════════════════════════════════
# Ticket 3 — deterministic filters
# ═══════════════════════════════════════════════════════════════════

def _contracts_df():
    return pd.DataFrame([
        {"id": "c1", "IsRisksAccepted": "no", "FlagNeedLegal": "yes", "Over5M": "yes"},
        {"id": "c2", "IsRisksAccepted": "yes", "FlagNeedLegal": "no", "Over5M": "no"},
        {"id": "c3", "IsRisksAccepted": "no", "FlagNeedLegal": "no", "Over5M": "no"},
    ])


def test_apply_risk_filters_exact_match():
    df = _contracts_df()
    out = apply_risk_filters(df, [{"field": "IsRisksAccepted", "op": "=", "value": "no"}])
    assert out["id"].tolist() == ["c1", "c3"]
    out2 = apply_risk_filters(df, [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}])
    assert out2["id"].tolist() == ["c1"]


def test_apply_risk_filters_combined_and_skips_missing_columns():
    df = _contracts_df()
    out = apply_risk_filters(df, [
        {"field": "IsRisksAccepted", "op": "=", "value": "no"},
        {"field": "NonExistentColumn", "op": "=", "value": "yes"},  # skipped
    ])
    assert out["id"].tolist() == ["c1", "c3"]


# ═══════════════════════════════════════════════════════════════════
# Ticket 4 — weighted scoring + severity
# ═══════════════════════════════════════════════════════════════════

def test_score_risk_weights_and_severity():
    df = _contracts_df()
    scored = score_risk(df)
    by_id = scored.set_index("id")
    # c1: IsRisksAccepted=no (50) + FlagNeedLegal=yes (20) + Over5M=yes (10) = 80 → high
    assert by_id.loc["c1", "risk_score"] == 80
    assert by_id.loc["c1", "risk_severity"] == "high"
    # c2: no signals → 0 → low
    assert by_id.loc["c2", "risk_score"] == 0
    assert by_id.loc["c2", "risk_severity"] == "low"
    # c3: IsRisksAccepted=no (50) = 50 → high (>= 50 threshold)
    assert by_id.loc["c3", "risk_score"] == 50
    assert by_id.loc["c3", "risk_severity"] == "high"
    assert "IsRisksAccepted = no (+50)" in by_id.loc["c1", "matched_signals"]
    assert "high" in by_id.loc["c1", "risk_explanation"]


def test_score_risk_medium_tier():
    df = pd.DataFrame([{"id": "m1", "FlagNeedLegal": "yes"}])  # 20 → medium
    scored = score_risk(df)
    assert scored.loc[0, "risk_score"] == 20
    assert scored.loc[0, "risk_severity"] == "medium"


def test_score_risk_custom_weights_override(monkeypatch):
    monkeypatch.setenv("RISK_WEIGHTS_JSON", '{"IsRisksAccepted|no": 99}')
    df = pd.DataFrame([{"id": "x", "IsRisksAccepted": "no"}])
    scored = score_risk(df)
    assert scored.loc[0, "risk_score"] == 99


# ═══════════════════════════════════════════════════════════════════
# Ticket 5 — gate + sort
# ═══════════════════════════════════════════════════════════════════

def test_gate_and_sort_filters_and_orders():
    df = _contracts_df()
    scored = score_risk(df)
    out = gate_and_sort(scored, min_score=20)
    assert out["id"].tolist() == ["c1", "c3"]  # c2 (0) gated out; c1 (80) before c3 (50)
    assert out["risk_score"].tolist() == [80, 50]


def test_gate_default_uses_medium_threshold():
    df = pd.DataFrame([
        {"id": "low1", "Over5M": "yes"},          # 10 → low, below gate
        {"id": "med1", "FlagNeedLegal": "yes"},   # 20 → medium, passes gate
    ])
    scored = score_risk(df)
    out = gate_and_sort(scored)  # default gate = medium threshold (20)
    assert out["id"].tolist() == ["med1"]


# ═══════════════════════════════════════════════════════════════════
# Ticket 2 — planner intent detection (no real LLM in tests)
# ═══════════════════════════════════════════════════════════════════

def test_planner_keyword_fallback_risky():
    planner = RiskPlanner()
    plan = planner.plan("show contracts where risk was not accepted")
    assert plan["mode"] == MODE_RISKY
    assert plan["confidence"] >= 0.6
    assert {"field": "IsRisksAccepted", "op": "=", "value": "no"} in plan["filters"]


def test_planner_keyword_fallback_legal_and_over5m():
    planner = RiskPlanner()
    plan = planner.plan("contracts needing legal review over 5m")
    assert plan["mode"] == MODE_RISKY
    fields = {c["field"] for c in plan["filters"]}
    assert "FlagNeedLegal" in fields and "Over5M" in fields


def test_planner_empty_query_clarifies():
    planner = RiskPlanner()
    plan = planner.plan("")
    assert plan["mode"] == MODE_CLARIFY
    assert plan.get("clarification_question")


def test_planner_llm_low_confidence_clarifies(monkeypatch):
    planner = RiskPlanner()
    monkeypatch.setattr(planner, "_call_llm", lambda *a, **k:
        '{"mode": "risky_contracts", "confidence": 0.4, "filters": [], "explanation": "unsure"}')
    plan = planner.plan("some ambiguous query with no keywords")
    assert plan["mode"] == MODE_CLARIFY
    assert plan.get("clarification_question") == DEFAULT_CLARIFICATION


def test_planner_llm_high_confidence_risky(monkeypatch):
    planner = RiskPlanner()
    monkeypatch.setattr(planner, "_call_llm", lambda *a, **k:
        '{"mode": "risky_contracts", "confidence": 0.9, "filters": '
        '[{"field": "FlagNeedLegal", "value": "yes"}, {"field": "BadField", "value": "yes"}], '
        '"explanation": "legal risk"}')
    plan = planner.plan("unusual phrasing about legal exposure")
    assert plan["mode"] == MODE_RISKY
    # BadField rejected by allowlist validation
    assert plan["filters"] == [{"field": "FlagNeedLegal", "op": "=", "value": "yes"}]


def test_planner_llm_failure_clarifies(monkeypatch):
    # LLM down → no confident routing → safest behaviour is to clarify with the user.
    planner = RiskPlanner()
    monkeypatch.setattr(planner, "_call_llm", lambda *a, **k: "")
    plan = planner.plan("renewal contract for acme")  # no risk keywords either
    assert plan["mode"] == MODE_CLARIFY
    assert plan.get("clarification_question")


def test_planner_llm_general_search(monkeypatch):
    planner = RiskPlanner()
    monkeypatch.setattr(planner, "_call_llm", lambda *a, **k:
        '{"mode": "general_search", "confidence": 0.9, "filters": [], "explanation": "plain search"}')
    plan = planner.plan("renewal contract for acme")
    assert plan["mode"] == MODE_GENERAL


# ═══════════════════════════════════════════════════════════════════
# End-to-end + UI smoke
# ═══════════════════════════════════════════════════════════════════

def test_run_risk_search_end_to_end():
    df = _contracts_df()
    plan = {"mode": MODE_RISKY, "confidence": 0.9,
            "filters": [{"field": "IsRisksAccepted", "op": "=", "value": "no"}]}
    out = run_risk_search(df, plan)
    assert out["id"].tolist() == ["c1", "c3"]
    assert {"risk_score", "risk_severity", "matched_signals", "risk_explanation"} <= set(out.columns)


def _make_agentic_st(records=None, buttons=None):
    """Minimal fake Streamlit for the unified agentic UI."""
    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    records = records if records is not None else {}
    st = types.ModuleType("streamlit")
    st.session_state = _FakeSessionState()
    st.subheader = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.json = lambda *a, **k: None
    st.text_input = lambda *a, **k: k.get("value", "")
    st.selectbox = lambda *a, **k: 0
    st.number_input = lambda *a, **k: k.get("value", 20)
    st.checkbox = lambda *a, **k: False
    st.spinner = lambda *a, **k: _Ctx()
    st.columns = lambda n, *a, **k: [_Ctx() for _ in range(
        len(n) if isinstance(n, (list, tuple)) else n)]
    st.warning = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.success = lambda *a, **k: None
    st.markdown = lambda text, *a, **k: records.setdefault("md", []).append(text)
    st.metric = lambda *a, **k: None
    st.expander = lambda *a, **k: _Ctx()
    st.dataframe = lambda df, *a, **k: records.setdefault("df", []).append(df)
    st.empty = lambda *a, **k: types.SimpleNamespace(
        markdown=lambda *a, **k: None, empty=lambda: None)
    st.button = buttons or (lambda label=None, *a, **k: bool(label and "Ask" in str(label)))
    return st


# ═══════════════════════════════════════════════════════════════════
# Ticket 9 — expanded risk tags + click-to-summarize
# ═══════════════════════════════════════════════════════════════════

def test_allowlist_expanded_with_new_tags():
    from apps.risk_search import RISK_FILTER_FIELDS
    new_tags = [
        "Over100M", "WithEndDate", "Saved", "unlimitedliabilitiesorliabilit",
        "relatedtocapexpropertyleasingc", "generalpurchaseandoverhk50k",
        "preliminaryreviewflag2", "needapreliminaryreviewbygroupl",
    ]
    for tag in new_tags:
        assert tag in RISK_FILTER_FIELDS


def test_keyword_fallback_new_tags():
    planner = RiskPlanner()
    plan = planner.plan("contracts with unlimited liability")
    assert plan["mode"] == MODE_RISKY
    fields = {c["field"] for c in plan["filters"]}
    assert "unlimitedliabilitiesorliabilit" in fields

    plan2 = planner.plan("any contracts related to capex or property leasing")
    fields2 = {c["field"] for c in plan2["filters"]}
    assert "relatedtocapexpropertyleasingc" in fields2

    plan3 = planner.plan("contracts including external guarantees")
    fields3 = {c["field"] for c in plan3["filters"]}
    assert "IncludingExternalGuarantees" in fields3


def test_score_risk_new_weighted_tags():
    from apps.risk_search import score_risk
    df = pd.DataFrame([{
        "id": "x1",
        "unlimitedliabilitiesorliabilit": "yes",   # +15
        "relatedtocapexpropertyleasingc": "yes",   # +10
        "generalpurchaseandoverhk50k": "yes",      # +10
    }])
    scored = score_risk(df)
    assert scored.loc[0, "risk_score"] == 35
    assert scored.loc[0, "risk_severity"] == "medium"


def test_build_contract_context_groups_tags():
    from apps.risk_search import build_contract_context
    row = pd.Series({
        "ref_no": "REF-1", "title": "Test Contract",
        "counterparty_name": "Acme", "department": "IT",
        "IfRelatedToData": "yes",
        "unlimitedliabilitiesorliabilit": "yes",
        "Over5M": "yes", "Over100M": "no",
        "risk_score": 40, "risk_severity": "medium",
        "matched_signals": ["IfRelatedToData = yes (+15)"],
    })
    ctx = build_contract_context(row)
    assert ctx["metadata"]["ref_no"] == "REF-1"
    assert "Related-party / data / capex" in ctx["risk_tags"]
    rel = ctx["risk_tags"]["Related-party / data / capex"]
    assert rel["involves personal/business data"] == "yes"
    assert rel["unlimited liability exposure"] == "yes"
    assert ctx["risk_score"] == 40
    assert ctx["matched_signals"] == ["IfRelatedToData = yes (+15)"]


def test_summarize_contract_deterministic_fallback():
    from apps.risk_search import summarize_contract
    row = pd.Series({
        "ref_no": "REF-9", "title": "Risky Deal",
        "counterparty_name": "Globex", "amount_label": "5,300,000.00",
        "IsRisksAccepted": "no", "unlimitedliabilitiesorliabilit": "yes",
        "risk_score": 65, "risk_severity": "high",
        "matched_signals": ["IsRisksAccepted = no (+50)", "unlimitedliabilitiesorliabilit = yes (+15)"],
    })
    result = summarize_contract(row, use_llm=False)
    assert result["llm_used"] is False
    text = result["summary"]
    assert "high" in text and "65" in text
    assert "tracked closely" in text or "re-examined" in text
    assert "legal review judgment" in text
    assert "\n\n" not in text


def test_summarize_contract_emphasizes_review_needed():
    from apps.risk_search import summarize_contract
    row = pd.Series({
        "ref_no": "REF-10", "title": "High Risk Deal",
        "counterparty_name": "Globex", "department": "Legal",
        "IsRisksAccepted": "no", "unlimitedliabilitiesorliabilit": "yes",
        "FlagNeedLegal": "yes", "risk_score": 95, "risk_severity": "high",
        "matched_signals": [
            "IsRisksAccepted = no (+50)",
            "FlagNeedLegal = yes (+10)",
            "unlimitedliabilitiesorliabilit = yes (+15)",
        ],
    })
    result = summarize_contract(row, use_llm=False)
    text = result["summary"]
    assert "tracked closely" in text.lower() or "re-examined" in text.lower()
    assert "legal review judgment" in text.lower()
    # Keep it paragraph-like even in fallback mode.
    assert "\n\n" not in text


def test_summarize_contract_llm_only_gets_context(monkeypatch):
    from apps.risk_search import summarize_contract, RiskPlanner
    captured = {}

    planner = RiskPlanner()

    def fake_llm(messages, max_tokens=400):
        captured["user"] = messages[1]["content"]
        return "LLM summary text."

    monkeypatch.setattr(planner, "_call_llm", fake_llm)
    row = pd.Series({
        "ref_no": "REF-LLM", "IfRelatedToData": "yes",
        "risk_score": 15, "risk_severity": "low",
        "matched_signals": ["IfRelatedToData = yes (+15)"],
    })
    result = summarize_contract(row, planner=planner, use_llm=True)
    assert result["summary"] == "LLM summary text."
    assert result["llm_used"] is True
    # LLM receives the explicit context (risk tags) and nothing else
    assert "IfRelatedToData" in captured["user"] or "involves personal/business data" in captured["user"]
    assert "REF-LLM" in captured["user"]
