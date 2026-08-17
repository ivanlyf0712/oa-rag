"""Deterministic natural-language -> contract filter extraction.

Single deep module owning the rule-based mapping from a user query to the
structured filter dict consumed by the contract search service. The LangChain
agent, the structured where_sql path, and the semantic post-filter all read
the same vocabulary here, so a label or alias change is made in one place.

Depth: one small interface (infer_contract_filters + the alias tables) hides
all of the per-field extraction rules. Status resolution delegates to the
shared status_labels module (the canonical DB-label vocabulary).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from apps.search.status_labels import STATUS_ALIASES

# Loose alias (lowercase) -> canonical DB contract-type label.
CONTRACT_TYPE_ALIASES: Dict[str, str] = {
    "nda": "NDA / Confidentiality Agreement",
    "confidentiality": "NDA / Confidentiality Agreement",
    "non-disclosure": "NDA / Confidentiality Agreement",
    "mou": "MOU / LOI",
    "loi": "MOU / LOI",
    "letter of intent": "MOU / LOI",
    "memorandum": "MOU / LOI",
    "sales agreement": "Sales Agreement / Service Agreement / Quotation",
    "service agreement": "Sales Agreement / Service Agreement / Quotation",
    "quotation": "Sales Agreement / Service Agreement / Quotation",
    "distribution agreement": "Distribution Agreement / Dealership Agreement",
    "dealership": "Distribution Agreement / Dealership Agreement",
    "procurement": "Procurement Agreement / Quotation (BU Principal/Supplier Contract)",
    "procurement agreement": "Procurement Agreement / Quotation (BU Principal/Supplier Contract)",
    "functional purchase": "Procurement Agreement / Quotation (Functional Purchase)",
    "lease": "Lease or Rental Agreement",
    "rental": "Lease or Rental Agreement",
    "others": "Others",
}

DEPARTMENT_ALIASES: Dict[str, str] = {
    "it": "IT",
    "information technology": "IT",
    "legal": "Legal",
    "finance": "Finance",
    "procurement": "Procurement",
    "sales": "Sales",
    "hr": "HR",
    "human resources": "HR",
    "operations": "Operations",
}

# Status words used for rule-based matching (where_sql path). Lowercase forms
# of the DB status labels plus common aliases a user might type.
STATUS_WORDS = (
    "draft", "pending", "preliminary", "returned", "final draft",
    "approval", "rejected", "completed", "signed", "terminated", "active",
)


def infer_contract_filters(query: str) -> Dict[str, Any]:
    """Deterministically extract contract facet filters from a query.

    Returns a plain dict of structured filters (status, contract_type,
    expired, contract_id, department, counterparty_name) -- empty when the
    query carries no extractable facet. Pure function; no I/O, no LLM.
    """
    text = (query or "").strip()
    lower = text.lower()
    filters: Dict[str, Any] = {}
    if not lower:
        return filters

    for needle, canonical in STATUS_ALIASES.items():
        if re.search(rf"\b{re.escape(needle)}\b", lower):
            filters["status"] = canonical
            break

    for needle, canonical in CONTRACT_TYPE_ALIASES.items():
        if re.search(rf"\b{re.escape(needle)}\b", lower):
            filters["contract_type"] = canonical
            break

    if re.search(r"\bexpired\b", lower):
        filters["expired"] = True

    m = re.search(
        r"\b(?:contract\s*id|id|ref\s*no|reference\s*no)\b[:\s#-]*([A-Za-z0-9][A-Za-z0-9/_-]*)",
        lower,
    )
    if m:
        filters["contract_id"] = m.group(1)

    m = re.search(r"\bdepartment\s+([A-Za-z][A-Za-z0-9/& -]*)", lower)
    if m:
        filters["department"] = m.group(1).strip()
    else:
        for needle, canonical in DEPARTMENT_ALIASES.items():
            if re.search(rf"\b{re.escape(needle)}\b", lower):
                filters["department"] = canonical
                break

    m = re.search(r"\bcounterparty\s+([A-Za-z][A-Za-z0-9/& ,.-]*)", lower)
    if m:
        filters["counterparty_name"] = m.group(1).strip()
    else:
        for kw in ("from", "with", "for"):
            m = re.search(rf"\b{kw}\s+([A-Za-z][A-Za-z0-9/& ,.-]*)", lower)
            if m:
                filters["counterparty_name"] = m.group(1).strip()
                break

    return filters


# Back-compat alias: the canonical status alias table lives in status_labels.
STATUS_ALIAS_TABLE = STATUS_ALIASES
