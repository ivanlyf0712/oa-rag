"""Tests for ticket 03: Ask UI merged table helpers.

Covers the pure table logic (flattening, min-score filter, risk ranking,
column picker) plus the calibrated min-score default. Widget wiring itself is
covered by the store render test in tests/test_observation_store.py.

Run:
    venv/bin/python -m pytest tests/test_ask_ui_table.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import apps.app as app_module
from apps.app import (
    DEFAULT_MIN_RISK_SCORE,
    _TABLE_BASE_COLUMNS,
    _TABLE_EXTRA_COLUMNS,
    _apply_table_controls,
    _flatten_result_rows,
)


def _rows():
    """Three contracts with descending risk scores (90 / 80 / 30)."""
    return [
        {"id": "a", "text": "liability text", "score": 0.7,
         "metadata": {"ref_no": "R-HIGH", "counterparty_name": "Alpha",
                      "contract_type": "2", "status_label": "active",
                      "contract_start_date": "2024-01-01",
                      "contract_end_date": "2025-01-01",
                      "amount_label": "HK$6M",
                      "risk_score": 90, "risk_severity": "high",
                      "matched_signals": ["IsRisksAccepted = no (+50)"],
                      "department": "IT"}},
        {"id": "b", "text": "renewal text", "score": 0.9,
         "metadata": {"ref_no": "R-EDGE", "counterparty_name": "Beta",
                      "risk_score": 80, "risk_severity": "high"}},
        {"id": "c", "text": "calm text", "score": 0.95,
         "metadata": {"ref_no": "R-LOW", "counterparty_name": "Gamma",
                      "risk_score": 30, "risk_severity": "medium"}},
    ]


# -- flattening -------------------------------------------------------------

def test_flatten_pulls_metadata_and_snippet():
    flat = _flatten_result_rows(_rows())
    assert len(flat) == 3
    high = flat[0]
    assert high["ref_no"] == "R-HIGH"
    assert high["counterparty_name"] == "Alpha"
    assert high["risk_score"] == 90
    assert high["risk_severity"] == "high"
    assert high["contract_start_date"] == "2024-01-01"
    assert high["amount_label"] == "HK$6M"
    assert "IsRisksAccepted = no (+50)" in high["matched_signals"]
    assert high["snippet"] == "liability text"


def test_min_score_default_is_calibrated_80():
    assert DEFAULT_MIN_RISK_SCORE == 80


# -- controls ---------------------------------------------------------------

def test_min_score_80_filters_to_high_risk_only():
    flat = _flatten_result_rows(_rows())
    df = _apply_table_controls(flat, rank_by_risk=False, min_score=80,
                               columns=_TABLE_BASE_COLUMNS)
    assert list(df["ref_no"]) == ["R-HIGH", "R-EDGE"]  # 80 is inclusive
    assert "R-LOW" not in list(df["ref_no"])


def test_min_score_zero_shows_everything():
    flat = _flatten_result_rows(_rows())
    df = _apply_table_controls(flat, rank_by_risk=False, min_score=0,
                               columns=_TABLE_BASE_COLUMNS)
    assert len(df) == 3


def test_rank_by_risk_reorders_descending():
    flat = _flatten_result_rows(_rows())
    # retrieval order has R-LOW first by relevance (score 0.95)
    rows = list(reversed(_rows()))
    flat = _flatten_result_rows(rows)
    df = _apply_table_controls(flat, rank_by_risk=True, min_score=0,
                               columns=_TABLE_BASE_COLUMNS)
    assert list(df["risk_score"]) == [90, 80, 30]


def test_relevance_keeps_retrieval_order():
    rows = list(reversed(_rows()))
    flat = _flatten_result_rows(rows)
    df = _apply_table_controls(flat, rank_by_risk=False, min_score=0,
                               columns=_TABLE_BASE_COLUMNS)
    assert list(df["ref_no"]) == ["R-LOW", "R-EDGE", "R-HIGH"]


def test_column_picker_selects_and_orders_columns():
    flat = _flatten_result_rows(_rows())
    df = _apply_table_controls(
        flat, rank_by_risk=False, min_score=0,
        columns=["ref_no", "department", "risk_score"])
    assert list(df.columns) == ["ref_no", "department", "risk_score"]
    assert df.iloc[0]["department"] == "IT"


def test_unknown_columns_are_dropped_and_empty_set_falls_back():
    flat = _flatten_result_rows(_rows())
    df = _apply_table_controls(flat, rank_by_risk=False, min_score=0,
                               columns=["ref_no", "bogus_col"])
    assert list(df.columns) == ["ref_no"]
    df = _apply_table_controls(flat, rank_by_risk=False, min_score=0, columns=[])
    assert list(df.columns) == list(_TABLE_BASE_COLUMNS)


def test_empty_result_set_yields_empty_table_with_columns():
    df = _apply_table_controls([], rank_by_risk=True, min_score=80,
                               columns=_TABLE_BASE_COLUMNS)
    assert df.empty
    assert list(df.columns) == list(_TABLE_BASE_COLUMNS)


def test_base_columns_cover_spec_and_extras_exist():
    # spec: ref, counterparty/title, type, status, dates, amount, risk
    for col in ("ref_no", "counterparty_name", "contract_type",
                "status", "amount_label", "risk_score", "risk_severity"):
        assert col in _TABLE_BASE_COLUMNS
    for col in ("matched_signals", "risk_explanation", "snippet"):
        assert col in _TABLE_EXTRA_COLUMNS
