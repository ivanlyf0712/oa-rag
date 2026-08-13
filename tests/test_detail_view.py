"""Tests for ticket 04: detail tabs Raw / Contextual / Risk helpers.

Run:
    venv/bin/python -m pytest tests/test_detail_view.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.detail_view import (
    CONTEXTUAL_GROUPS,
    EMPTY_PLACEHOLDER,
    build_contextual_groups,
    coalesce_raw,
    humanize_signal,
    humanize_signals,
)
from core.db import BOOLEAN_CODE_FIELDS


def _rich_record():
    return {
        "ref_no": "CCA20250096",
        "title": "IT Upgrade Agreement",
        "counterparty_name": "Alpha Corp",
        "department": "IT",
        "contract_type": "2",
        "contract_start_date": "2024-01-01",
        "contract_end_date": "2025-12-31",
        "requested_date": "2023-12-15",
        "amount_label": "HK$6,000,000",
        "status_label": "active",
        "contextual_fields": {
            "contractowner": "Jane Chan",
            "BusinessApprovalLevel": "L2",
            "NoteOnContractAmount": "includes hardware",
        },
        "decoded_fields": {
            "IsRisksAccepted": {"raw": 0, "label": "no"},
            "Over5M": {"raw": 1, "label": "yes"},
            "FlagNeedLegal": {"raw": 2, "label": "na"},
        },
    }


# -- contextual groups ------------------------------------------------------

def test_contextual_groups_render_curated_fields():
    groups = build_contextual_groups(_rich_record())
    by_group = {g["group"]: {f["label"]: f["value"] for f in g["fields"]}
                for g in groups}
    identity = by_group["Identity & lifecycle"]
    assert identity["Reference No"] == "CCA20250096"
    assert identity["Counterparty"] == "Alpha Corp"
    assert identity["Start date"] == "2024-01-01"
    assert by_group["Approval & workflow state"]["Status"] == "active"
    assert by_group["Ownership & routing"]["Contract owner"] == "Jane Chan"


def test_sparse_record_renders_placeholders_everywhere():
    groups = build_contextual_groups({"ref_no": "R1"})
    for group in groups:
        assert group["fields"]  # every group still renders
    identity = {f["label"]: f["value"] for f in groups[0]["fields"]}
    assert identity["Reference No"] == "R1"
    assert identity["Title"] == EMPTY_PLACEHOLDER
    assert identity["End date"] == EMPTY_PLACEHOLDER


def test_coded_fields_render_decoded_labels():
    groups = build_contextual_groups(_rich_record())
    coded = {f["label"]: f["value"] for f in groups[-1]["fields"]}
    assert coded["risks accepted"] == "No"
    assert coded["amount over 5M"] == "Yes"
    assert coded["legal review required"] == "N/A"
    # every coded field is represented (present or placeholder)
    assert len(groups[-1]["fields"]) == len(BOOLEAN_CODE_FIELDS)


def test_undecodable_codes_excluded_from_contextual():
    rec = _rich_record()
    rec["decoded_fields"]["Over5M"] = {"raw": "STRANGE", "label": None}
    groups = build_contextual_groups(rec)
    coded_labels = [f["label"] for f in groups[-1]["fields"]]
    assert "amount over 5M" not in coded_labels  # excluded, stays in Raw


def test_audit_and_system_fields_not_in_contextual_groups():
    all_keys = [k for _, fields in CONTEXTUAL_GROUPS for _, keys in fields for k in keys]
    for opaque in ("MODEUUID", "formmodeid", "modedatacreater", "processId",
                   "modedatamodifydatetime", "constactid", "orgprocess"):
        assert opaque not in all_keys


# -- risk signals humanization ----------------------------------------------

def test_humanize_signal_uses_field_labels():
    assert humanize_signal("Over5M = yes (+25)") == "amount over 5M = yes (+25)"
    assert humanize_signal("IsRisksAccepted = no (+50)") == \
        "risks accepted = no (+50)"
    # unknown fields pass through unchanged
    assert humanize_signal("SomeOtherField = yes (+1)") == "SomeOtherField = yes (+1)"


def test_humanize_signals_handles_empty():
    assert humanize_signals([]) == []
    assert humanize_signals(None) == []


# -- raw coalescing -----------------------------------------------------------

def test_coalesce_raw_prefers_full_raw_record():
    raw = {"RefNo": "R1", "SomeColumn": "x"}
    record, is_full = coalesce_raw(raw, {"ref_no": "R1"})
    assert is_full is True
    assert record is raw


def test_coalesce_raw_falls_back_to_stored_metadata():
    record, is_full = coalesce_raw(None, {"ref_no": "R1", "empty": None, "blank": ""})
    assert is_full is False
    assert record == {"ref_no": "R1"}  # empty values dropped


def test_coalesce_raw_handles_stringified_dict():
    record, is_full = coalesce_raw("not-a-dict", {"a": 1})
    assert is_full is False and record == {"a": 1}
