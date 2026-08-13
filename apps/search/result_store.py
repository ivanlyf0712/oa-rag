"""Per-invocation result store for the unified search pipeline.

The unified search tool adapter stashes its full ContractRow set here; the
UI reads a snapshot instead of re-querying or parsing the LLM observation
text (the old regex observation parser is deleted). Pattern follows
corpchat-rag's snapshot_meta: a module-level dict guarded by a lock; the UI
snapshots immediately after agent.process() returns (same thread), so each
tool call is attributed its own results.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()

_EMPTY: Dict[str, Any] = {
    "rows": [],
    "query": "",
    "filters": {},
    "total": 0,
    "rank_by": "relevance",
    "observation_count": 0,
}

_last: Dict[str, Any] = dict(_EMPTY)


def stash_results(
    rows: Optional[List[Dict[str, Any]]],
    *,
    query: str = "",
    filters: Optional[Dict[str, Any]] = None,
    rank_by: str = "relevance",
    observation_count: Optional[int] = None,
) -> None:
    """Replace the store with the results of the latest tool invocation."""
    rows = list(rows or [])
    with _LOCK:
        _last.clear()
        _last.update({
            "rows": rows,
            "query": query or "",
            "filters": dict(filters or {}),
            "total": len(rows),
            "rank_by": rank_by or "relevance",
            "observation_count": (
                observation_count if observation_count is not None else len(rows)
            ),
        })


def snapshot_results() -> Dict[str, Any]:
    """Return a copy of the latest stashed results (query, rows, meta)."""
    with _LOCK:
        snap = dict(_last)
        snap["rows"] = list(_last.get("rows") or [])
        snap["filters"] = dict(_last.get("filters") or {})
        return snap


def clear_results() -> None:
    """Reset the store (called before each new agent turn)."""
    with _LOCK:
        _last.clear()
        _last.update(_EMPTY)
