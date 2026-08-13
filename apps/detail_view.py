"""Contract detail view helpers (ticket 04) — pure, unit-tested.

The detail view is three lenses over one stored record:
  - Raw: the complete stored record as JSON (no curation).
  - Contextual: curated self-explanatory fields + decoded Yes/No labels for
    coded fields, grouped per data/contract-data-column-groups.md; audit /
    system identifiers and undecodable codes are excluded (they stay in Raw);
    missing values render an explicit "(empty)" placeholder.
  - Risk: score/severity/labeled signals derived from the same normalized
    record (single source of truth).

All helpers are Streamlit-free so they can be unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from apps.risk_search import _FIELD_LABELS
from core.db import BOOLEAN_CODE_FIELDS

EMPTY_PLACEHOLDER = "(empty)"

# Curated contextual fields, grouped per data/contract-data-column-groups.md.
# Each entry: (display label, (candidate keys in priority order)).
# Candidates are looked up first in the flat metadata, then in
# contextual_fields. Audit/system identifiers (modedatacreater, MODEUUID,
# formmodeid, processId, ...) are deliberately excluded — they remain in Raw.
CONTEXTUAL_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, Tuple[str, ...]], ...]], ...] = (
    ("Identity & lifecycle", (
        ("Reference No", ("ref_no", "RefNo")),
        ("Title", ("title", "TitleReferenceNoOfContract")),
        ("Counterparty", ("counterparty_name", "CounterpartyName")),
        ("Department", ("department", "businessunit")),
        ("Contract type", ("contract_type", "contracttype")),
        ("Start date", ("contract_start_date", "contractstartdate")),
        ("End date", ("contract_end_date", "contractenddate")),
        ("Requested date", ("requested_date",)),
    )),
    ("Commercial terms", (
        ("Amount (HKD)", ("amount_label", "ContractAmountHKD")),
        ("Note on amount", ("NoteOnContractAmount",)),
        ("Revenue (year 1 / total / previous)",
         ("revenueyear1", "revenuetotal", "revenueprevious")),
        ("Gross profit (year 1 / total / previous)",
         ("gpyear1", "gptotal", "gpprevious")),
        ("NPAT (year 1 / total / previous)",
         ("npatyear1", "npattotal", "npatprevious")),
    )),
    ("Ownership & routing", (
        ("Contract owner", ("contractowner",)),
        ("Entity contract owner", ("entitycontractowner",)),
        ("Requestor", ("requestor",)),
        ("Business unit", ("businessunit",)),
        ("Requested business unit", ("requestedbusinessunit",)),
        ("Signing entity", ("dchsigningentity1",)),
    )),
    ("Approval & workflow state", (
        ("Status", ("status_label", "Status", "status")),
        ("Business approval level", ("BusinessApprovalLevel",)),
        ("Finance approval level", ("FinanceApprovalLevel",)),
        ("Detail finance approval level", ("DetailFinanceApprovalLevel",)),
        ("Matrix finance approval level", ("MatrixFinanceApprovalLevel",)),
        ("Sign-off level", ("SignoffLevel",)),
        ("Display level", ("DisplayLevel",)),
        ("Review tier", ("reviewtier",)),
        ("Risk endorser", ("RiskEndorser",)),
    )),
)

_DECODED_LABEL_DISPLAY = {"yes": "Yes", "no": "No", "na": "N/A"}


def _lookup(meta: Dict[str, Any], contextual: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        value = meta.get(key)
        if value is not None and str(value).strip() != "":
            return value
        value = contextual.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _display_label(value: Any) -> str:
    text = str(value).strip().lower()
    return _DECODED_LABEL_DISPLAY.get(text, str(value))


def build_contextual_groups(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build grouped Contextual-tab display data from one normalized record.

    `record` is the flat metadata dict (service row metadata or a
    _load_sections row) carrying decoded_fields / contextual_fields.
    Returns [{"group": str, "fields": [{"label", "value"}]}, ...] where every
    curated field is present (missing -> "(empty)") and a final group lists
    the decoded Yes/No labels for the coded fields. Coded fields whose raw
    value cannot be decoded are excluded (visible in Raw instead).
    """
    contextual = record.get("contextual_fields") or {}
    decoded = record.get("decoded_fields") or {}

    groups: List[Dict[str, Any]] = []
    for group_name, fields in CONTEXTUAL_GROUPS:
        rendered = []
        for label, keys in fields:
            value = _lookup(record, contextual, keys)
            rendered.append({
                "label": label,
                "value": str(value) if value is not None else EMPTY_PLACEHOLDER,
            })
        groups.append({"group": group_name, "fields": rendered})

    coded = []
    for field in BOOLEAN_CODE_FIELDS:
        pair = decoded.get(field)
        if not pair:
            coded.append({
                "label": _FIELD_LABELS.get(field, field),
                "value": EMPTY_PLACEHOLDER,
            })
            continue
        label = pair.get("label")
        if label is None and pair.get("raw") not in (None, ""):
            continue  # undecodable code -> excluded from Contextual (kept in Raw)
        coded.append({
            "label": _FIELD_LABELS.get(field, field),
            "value": _display_label(label) if label is not None else EMPTY_PLACEHOLDER,
        })
    groups.append({"group": "Coded risk / policy fields (decoded)", "fields": coded})
    return groups


def humanize_signal(signal: str) -> str:
    """Rewrite 'Field = value (+pts)' with the human-readable field label."""
    text = str(signal)
    field = text.split(" = ", 1)[0].strip()
    human = _FIELD_LABELS.get(field)
    if not human or field == human:
        return text
    return text.replace(field, human, 1)


def humanize_signals(signals: List[str]) -> List[str]:
    return [humanize_signal(s) for s in (signals or [])]


def coalesce_raw(raw: Any, stored: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Return (record_for_raw_tab, is_complete_raw).

    The full `raw` record (~155 columns) is present once the index is rebuilt
    with raw in the chunk metadata. Until then, fall back to the complete
    stored metadata (every indexed field, no curation) and report False.
    """
    if isinstance(raw, dict) and raw:
        return raw, True
    cleaned = {
        k: v for k, v in (stored or {}).items()
        if v is not None and str(v) != "" and str(v).lower() != "nan"
    }
    return cleaned, False
