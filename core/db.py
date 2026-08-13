# ──────────────────── Database Module ────────────────────
from __future__ import annotations

import json
import os
import re
import warnings
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pymysql

from core.config import build_db_config

warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

DEFAULT_CONTRACT_TABLE = os.getenv("CONTRACTS_TABLE", "formtable_main_385")


def get_db_connection():
    return pymysql.connect(**build_db_config())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _compact_text(value: Any, limit: int = 1200) -> str:
    text = _coerce_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return bool(int(value))
    text = _coerce_text(value).strip().lower()
    if text in {"1", "true", "yes", "y", "approved", "ok", "pass"}:
        return True
    if text in {"0", "false", "no", "n", "rejected", "fail"}:
        return False
    return None


def _pick_first(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _pick_text(record: Dict[str, Any], *keys: str) -> str:
    value = _pick_first(record, *keys)
    return _coerce_text(value).strip() if value is not None else ""


def _parse_tags(record: Dict[str, Any]) -> List[str]:
    raw = _pick_first(record, "tags", "Tags", "tag", "keywords")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict):
        return [str(item).strip() for item in raw.values() if str(item).strip()]
    text = _coerce_text(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict):
            return [str(item).strip() for item in parsed.values() if str(item).strip()]
    except Exception:
        pass
    parts = re.split(r"[,;，|/]+", text)
    return [part.strip() for part in parts if part.strip()]


# ── Boolean / state code decoding ────────────────────────────────────
# The 385 form table uses 0/1/2 coded fields. Without decoding, the
# encoder/retriever has no idea what "1" means. We keep the raw value
# AND attach a human-readable label so the index is searchable by meaning.

_CODE_LABELS = {
    "0": "no",
    "1": "yes",
    "2": "na",
}

STATUS_LABELS = {
    0: "Draft",
    1: "Pending Preliminary Review",
    2: "Returned from Preliminary Review",
    3: "Pending Final Draft",
    4: "Pending Approval",
    5: "Rejected",
    6: "Pending Signed Contract",
    7: "Completed",
}


def _status_label(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return STATUS_LABELS.get(int(value), str(value))
    except Exception:
        return str(value)

# Fields in formtable_main_385 that use 0/1/2 (or 0/1) state codes.
# These get decoded into raw + label pairs. Fields using 0/1 only are
# booleans; fields that may contain 2 are ternary (yes/no/na).
# Note: fields verified to be NULL/empty across ALL rows in the live table
# (Solely, IsPreAuthoritySufficient, IsMC, PromptForOver5M,
# PromptForJustificationsUnder5M, PromptForRelatedToData, hideforcontractenddate,
# hidefornorenew, hideforcontractfinancial, hideforkeyrisks,
# hideforriskwarningsection) were removed — they never produce a value, so they
# cannot affect vectors or display. They remain available in record["raw"].
BOOLEAN_CODE_FIELDS = [
    # threshold / risk flags
    "Over5M", "Over100M", "WithEndDate", "Saved",
    "IncludingExternalGuarantees",
    "IsAuthoritySufficient",
    "IsRisksAccepted", "IsRenew", "iscontractfinancial",
    "needapreliminaryreviewbygroupl", "PreliminaryReviewFlag",
    "preliminaryreviewflag2",
    # need-approval flags
    "FlagNeedLegal", "FlagNeedGFN",
    # related-party / data / capex flags
    "IfRelatedToData", "relatedtocapexpropertyleasingc",
    "generalpurchaseandoverhk50k", "unlimitedliabilitiesorliabilit",
    # documentation completeness
    "allrelevantdocumentationhasbee",
    # acknowledgement
    "ihaveread10points",
]

# Fields whose values are workflow/display state but NOT 0/1/2 — keep raw.
# (Status, DisplayLevel, reviewtier, approval levels etc. stay as-is.)


def _decode_code(value: Any) -> Optional[str]:
    """Decode a 0/1/2 (or boolean-ish) coded value to no/yes/na.

    Returns None when the value is empty or not a recognized code.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = _coerce_text(value).strip().lower()
    if not text:
        return None
    if text in _CODE_LABELS:
        return _CODE_LABELS[text]
    if text in {"true", "y", "yes"}:
        return "yes"
    if text in {"false", "n", "no"}:
        return "no"
    if text in {"n/a", "na", "not applicable"}:
        return "na"
    return None


def _build_decoded_fields(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build raw+label pairs for every coded field present in the record."""
    decoded: Dict[str, Dict[str, Any]] = {}
    for field in BOOLEAN_CODE_FIELDS:
        raw = _pick_first(record, field)
        label = _decode_code(raw)
        if raw in (None, "") and label is None:
            continue
        decoded[field] = {
            "raw": _normalize_value(raw),
            "label": label,
        }
    return decoded


def _decoded_context_lines(decoded: Dict[str, Dict[str, Any]]) -> List[str]:
    """Render decoded coded fields as 'Field: label' lines for indexing."""
    lines: List[str] = []
    for field, pair in decoded.items():
        label = pair.get("label")
        if label:
            lines.append(f"{field}: {label}")
    return lines


def _extract_contract_text(record: Dict[str, Any]) -> str:
    # Free-text / narrative fields that carry contract meaning.
    candidate_fields = [
        "finalversioncontract",
        "signedcontract",
        "DraftContract",
        "detailedbudgetapprovalrecord",
        "description",
        "KeyChanges",
        "HistoricalVersions",
        "PreviousSignedContracts",
        "CounterpartyName_MultiLine",
        "bufinancerole",
        # justification / reason narratives
        "ReasonToSubmitLT5M",
        "ReasonsNoEndDa",
        "NoteOnContractAmount",
        "reasonsfornotuploadingallthedo",
        # risk assessment prompt answers (free text).
        # assessmentprompt2-6 are NULL across all rows (dropped); 1 & 7 have content.
        "assessmentprompt1", "assessmentprompt7",
    ]
    chunks: List[str] = []
    for key in candidate_fields:
        text = _pick_text(record, key)
        if text:
            chunks.append(text)
    if not chunks:
        fallback = _pick_text(record, "description", "TitleReferenceNoOfContract", "CounterpartyName", "ProductServices")
        if fallback:
            chunks.append(fallback)
    return "\n\n".join(chunks)


# Contextual raw fields worth surfacing to the index/UI even though they are
# not free-text narratives. Organized by the column groups in
# contract-data-column-groups.md. These are included verbatim (no
# interpretation) so reviewers and the retriever see source-of-truth values.
# Fields removed because they are NULL/empty across ALL rows in the live table:
# requestId, ProductServices, BUApprovalGrade, BusinessApproverSecurityLevel,
# BusinessApprover, BUFinanceApprover, GroupFinanceApprover, LegalApprover,
# BUName, DCHSigningEntity. (Still in record["raw"].)
CONTEXTUAL_FIELDS = [
    # 1) identity / lifecycle
    "RefNo", "TitleReferenceNoOfContract",
    "contractstartdate", "contractenddate",
    # 2) commercial terms (incl. all financial % metrics that have content)
    "ContractAmountHKD", "NoteOnContractAmount",
    "contracttype", "contract_type",
    "revenueyear1", "revenuetotal", "revenueprevious",
    "gpyear1", "gptotal", "gpprevious",
    "gpyear1percent", "gptotalpercent", "gppreviouspercent",
    "npatyear1", "npattotal", "npatprevious",
    "npatyear1percent", "npattotalpercent", "npatpreviouspercent",
    "roicyear1percent", "roictotalpercent", "roicpreviouspercent",
    # 5) approval routing / sign-off (levels, grades, final flags, L2/L3 chain)
    "Status", "BusinessApprovalLevel", "FinanceApprovalLevel",
    "DetailFinanceApprovalLevel", "MatrixFinanceApprovalLevel",
    "SignoffLevel", "DisplayLevel",
    "BusinessSecurityLevel",
    "FinalBusiness", "FinalBUFinance", "FinalGroupFinance", "FinalLegal",
    "RiskEndorser",
    "BUApprovalGradeL1", "BusinessSecurityL1", "BusinessApproverSecurityL1",
    "PreBusinessApprover", "FinalBusinessApprover",
    "l2businessapprover", "l3businessapprover",
    "level2businessapprover", "level3businessapprover",
    "l2financeapprover", "l3financeapprover",
    "level1financeapprover", "level2financeapprover",
    "BUFinanceApproverL1",
    # 6) ownership / routing (incl. committee/head references, which have content)
    "contractowner", "entitycontractowner", "requestor",
    "businessunit", "Department", "dchsigningentity1", "requestedbusinessunit",
    "entityfinanceheadlilist",
    "BuHead", "EntityFinanceHead", "BuGroupFinanceHead",
    "GroupCFO", "ManagementCommittee",
    # 7) workflow state
    "reviewtier", "requested_date", "requested_time",
    # 8) audit / system identifiers (all populated)
    "modedatacreater", "modedatacreatedate", "modedatacreatetime",
    "modedatamodifier", "modedatamodifydatetime",
    "formmodeid", "modedatacreatertype",
    "MODEUUID", "constactid", "orgprocess", "processId", "isDeleteProcess",
]


def _build_contextual_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract contextual (non-narrative) raw fields verbatim."""
    out: Dict[str, Any] = {}
    for field in CONTEXTUAL_FIELDS:
        value = _pick_first(record, field)
        if value not in (None, ""):
            out[field] = _normalize_value(value)
    return out


def _normalize_contract_record(record: Dict[str, Any]) -> Dict[str, Any]:
    amount_raw = _pick_first(record, "ContractAmountHKD", "amount", "Amount")
    amount = None
    if amount_raw is not None:
        amount_text = _coerce_text(amount_raw).replace(",", "").strip()
        try:
            amount = float(amount_text)
        except Exception:
            amount = _normalize_value(amount_raw)

    contract_text = _extract_contract_text(record)
    title = _pick_text(record, "TitleReferenceNoOfContract", "RefNo", "CounterpartyName", "ProductServices")
    counterparty = _pick_text(record, "CounterpartyName")
    department = _pick_text(record, "businessunit", "Department")
    department_id = _pick_first(record, "Department", "department")
    start_date = _pick_text(record, "contractstartdate")
    end_date = _pick_text(record, "contractenddate")
    requested_date = _pick_text(record, "requested_date")
    tags = _parse_tags(record)

    legal_approval = _to_bool(_pick_first(record, "FinalLegal", "LegalApprover", "FlagNeedLegal"))
    overruled = _to_bool(_pick_first(record, "IsRisksAccepted", "Saved", "PreliminaryReviewFlag", "PreliminaryReviewFlag2"))

    # De-normalized views: raw contextual fields + decoded coded fields.
    decoded_fields = _build_decoded_fields(record)
    contextual_fields = _build_contextual_fields(record)
    search_context = _decoded_context_lines(decoded_fields)

    return {
        "id": record.get("id"),
        "request_id": _pick_first(record, "requestId"),
        "title": title or f"Contract #{record.get('id')}",
        "counterparty_name": counterparty,
        "product_services": _pick_text(record, "ProductServices"),
        "department": department,
        "department_id": _normalize_value(department_id) if department_id is not None else None,
        "amount": amount,
        "amount_label": _pick_text(record, "ContractAmountHKD"),
        "contract_start_date": start_date,
        "contract_end_date": end_date,
        "requested_date": requested_date,
        "sign_date": start_date or requested_date,
        "sign_date_label": start_date or requested_date,
        "content": contract_text,
        "content_preview": _compact_text(contract_text, 300),
        "tags": tags,
        "legal_approval": legal_approval,
        "overruled": overruled,
        "status": _pick_first(record, "Status"),
        "status_label": _status_label(_pick_first(record, "Status")),
        "is_deleted": _pick_first(record, "isDeleteProcess"),
        "ref_no": _pick_text(record, "RefNo"),
        "contract_type": _pick_first(record, "contracttype", "contract_type"),
        # new de-normalized structures
        "decoded_fields": decoded_fields,
        "contextual_fields": contextual_fields,
        "search_context": search_context,
        "raw": {k: _normalize_value(v) for k, v in record.items()},
    }


def fetch_contracts(limit: int = 10000, table_name: str = DEFAULT_CONTRACT_TABLE) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    query = "SELECT * FROM {} LIMIT %s".format(table_name)
    cur.execute(query, (limit,))
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()

    contracts: List[Dict[str, Any]] = []
    for row in rows:
        # DictCursor returns dicts already; a plain cursor returns tuples.
        record = dict(row) if isinstance(row, dict) else dict(zip(columns, row))
        contracts.append(_normalize_contract_record(record))
    return contracts
