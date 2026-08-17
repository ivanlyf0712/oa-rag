"""Canonical contract-status vocabulary shared by the search stack.

Single source of truth for mapping loose user language ("done", "pending",
"awaiting approval") onto the DB status labels stored in contract metadata
(status_label). Both the semantic post-filter (service._normalize_status)
and the agent's rule-based filter extraction (LangChainAgent) read from
here, so a label change is made in exactly one place.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    from core.db import STATUS_LABELS as _DB_STATUS_LABELS
except ImportError:  # pragma: no cover - DB layer optional in some contexts
    _DB_STATUS_LABELS = {}

# Loose alias (lowercase) -> canonical DB label (matches core.db.STATUS_LABELS).
STATUS_ALIASES = {
    "completed": "Completed",
    "complete": "Completed",
    "done": "Completed",
    "finished": "Completed",
    "approved": "Completed",
    "draft": "Draft",
    "pending preliminary": "Pending Preliminary Review",
    "preliminary review": "Pending Preliminary Review",
    "returned preliminary": "Returned from Preliminary Review",
    "pending final draft": "Pending Final Draft",
    "final draft": "Pending Final Draft",
    "pending approval": "Pending Approval",
    "awaiting approval": "Pending Approval",
    "rejected": "Rejected",
    "terminated": "Rejected",
    "pending signed": "Pending Signed Contract",
    "signed contract": "Pending Signed Contract",
    # Ambiguous shorthands map to the closest real status.
    "pending": "Pending Approval",
    "active": "Pending Approval",
}


def normalize_status(value: Any) -> Optional[str]:
    """Resolve a user-supplied status to the canonical DB label.

    Returns the exact STATUS_LABELS value on an exact (case-insensitive)
    match, else the aliased canonical label, else the raw stripped text
    unchanged (so unknown statuses still filter by exact match downstream).
    None / empty -> None.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    # Exact DB label match (case-insensitive) keeps a raw label stable.
    for label in _DB_STATUS_LABELS.values():
        if label.lower() == lower:
            return label
    return STATUS_ALIASES.get(lower, text)
