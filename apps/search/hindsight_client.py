"""
OA Search — Hindsight Memory Bridge
====================================
REST client for the Hindsight memory layer (API :8888), ported from
corpchat-rag (apps/corpchat/search/hindsight_client.py).

Gives OA RAG a cross-session memory backend:
  - retain: store queries/evidence summaries into a memory bank
  - recall: retrieve relevant past memories for a query
  - reflect: generate an answer grounded in bank memory + directives
  - get_entity_graph: fetch the bank's entity graph (for UI preview)
  - get_disposition / set_disposition: the CARA personality bridge

All calls degrade gracefully (return empty/None on failure) so the search
UI keeps working when Hindsight is down.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests

_DEFAULT_URL = "http://localhost:8888"
_DEFAULT_BANK = "test-bank"


# ── On-demand recall gate (决策 16: 记忆触达词) ─────────────────────
# 仅命中显式跨会话引用词才需要 Hindsight recall; 裸指代词 (她/他/这个/那个)
# 刻意不触发 —— 会话内指代由会话历史注入解析, 触发 recall 只会浪费一次调用。
_MEMORY_TRIGGER_KEYWORDS = (
    # 中文: 显式跨会话引用
    "上次", "上回", "上上次", "之前", "以前", "记得", "还记得",
    "当时", "那次", "上次说", "上次聊", "上回说", "上回聊",
    "之前说", "之前聊", "以前说", "以前聊", "那件事",
    # 英文: explicit cross-session references
    "last time", "remember", "before", "previously", "earlier",
    "as i said", "we discussed",
)


def needs_recall(query: str) -> bool:
    """Hindsight 按需 recall 判定 (gate)。

    命中显式跨会话引用词 → True (需要注入历史记忆); 否则 False。
    - 中文关键词按边界匹配 (CJK 字符天然满足非 a-z 边界, 等价于子串匹配);
    - 英文多词短语按子串匹配, 单英文词按整词匹配 ("before" 不命中 "beforehand");
    - 刻意排除裸指代词, 避免 "她的邮箱" 这类会话内指代误触发。
    """
    ql = (query or "").lower().strip()
    for kw in _MEMORY_TRIGGER_KEYWORDS:
        if " " in kw:
            if kw in ql:
                return True
        elif re.search(rf"(^|[^a-z]){re.escape(kw)}([^a-z]|$)", ql):
            return True
    return False


def _base_url() -> str:
    return (os.getenv("HINDSIGHT_API_URL") or _DEFAULT_URL).rstrip("/")


def _bank_id() -> str:
    return os.getenv("HINDSIGHT_BANK_ID") or _DEFAULT_BANK


def _url(path: str, bank: Optional[str] = None) -> str:
    b = bank or _bank_id()
    return f"{_base_url()}/v1/default/banks/{b}{path}"


def _timeout() -> int:
    try:
        return int(os.getenv("HINDSIGHT_API_TIMEOUT", "10"))
    except ValueError:
        return 10


# ── Retain ───────────────────────────────────────────────────────
def retain(content: str, bank: Optional[str] = None, *,
           context: Optional[str] = None,
           document_id: Optional[str] = None,
           tags: Optional[List[str]] = None,
           async_: bool = False) -> bool:
    """Store a memory into the bank (best-effort).

    Returns True if Hindsight accepted the retain request.
    """
    try:
        item: Dict[str, Any] = {"content": content}
        if context:
            item["context"] = context
        if document_id:
            item["document_id"] = document_id
        if tags:
            item["tags"] = tags
        resp = requests.post(
            _url("/memories", bank),
            json={"items": [item], "async": async_},
            timeout=_timeout(),
        )
        return resp.status_code in (200, 201, 202)
    except Exception:
        return False


# ── Recall ───────────────────────────────────────────────────────
def recall(query: str, bank: Optional[str] = None, *,
           max_results: int = 5) -> List[Dict]:
    """Retrieve relevant past memories for a query.

    Returns a list of memory dicts, each with 'content' (and optional
    'score' / 'type' / 'context'). Empty list on failure.
    """
    try:
        resp = requests.post(
            _url("/memories/recall", bank),
            json={"query": query},
            params={"limit": max_results},
            timeout=_timeout(),
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        # 响应结构: {"results": [{id, text, type, entities, scores, ...}]}
        if isinstance(data, dict):
            results = data.get("results", data.get("items", data.get("memories", [])))
        else:
            results = data
        if isinstance(results, dict):
            results = results.get("items", [])
        out = []
        for r in results:
            if not isinstance(r, dict):
                continue
            # 统一 content 字段 (API 用 text, 保留兼容)
            if "content" not in r and "text" in r:
                r = dict(r, content=r["text"])
            out.append(r)
        return out[:max_results]
    except Exception:
        return []


# ── Reflect ──────────────────────────────────────────────────────
def reflect(query: str, bank: Optional[str] = None, *,
            max_tokens: int = 300) -> str:
    """Generate an answer grounded in bank memory + directives.

    Returns the assistant's text, or "" on failure.
    """
    try:
        resp = requests.post(
            _url("/reflect", bank),
            json={"query": query, "max_tokens": max_tokens},
            timeout=_timeout() + 10,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        if isinstance(data, dict):
            return data.get("answer") or data.get("content") or data.get("response") or ""
        return str(data)
    except Exception:
        return ""


# ── Entity graph (UI preview) ────────────────────────────────────
def get_entity_graph(bank: Optional[str] = None, *,
                     limit: int = 50) -> Dict:
    """Fetch the bank's entity graph for UI preview.

    Returns {"nodes": [...], "edges": [...], "total_entities": n,
    "total_edges": n} or empty dict on failure.
    """
    try:
        resp = requests.get(
            _url("/entities/graph", bank),
            params={"limit": limit},
            timeout=_timeout(),
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if not isinstance(data, dict):
            return {}
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        flat_nodes = [n.get("data", n) for n in nodes] if nodes else []
        flat_edges = [e.get("data", e) for e in edges] if edges else []
        return {
            "nodes": flat_nodes,
            "edges": flat_edges,
            "total_entities": data.get("total_entities", len(flat_nodes)),
            "total_edges": data.get("total_edges", len(flat_edges)),
        }
    except Exception:
        return {}


# ── Bank list (UI convenience) ───────────────────────────────────
def list_banks() -> List[str]:
    """List available memory banks. Empty list on failure."""
    try:
        resp = requests.get(f"{_base_url()}/v1/default/banks", timeout=_timeout())
        if resp.status_code != 200:
            return []
        data = resp.json()
        banks = data.get("banks", []) if isinstance(data, dict) else []
        return [b.get("bank_id", "") for b in banks if isinstance(b, dict) and b.get("bank_id")]
    except Exception:
        return []


# ── Disposition (CARA bridge) ────────────────────────────────────
def get_disposition(bank: Optional[str] = None) -> Dict:
    """Read the bank's disposition traits (1-5 each). Empty dict on failure."""
    try:
        resp = requests.get(_url("/config", bank), timeout=_timeout())
        if resp.status_code != 200:
            return {}
        cfg = resp.json().get("config", {})
        return {
            "skepticism": cfg.get("disposition_skepticism"),
            "literality": cfg.get("disposition_literalism"),
            "empathy": cfg.get("disposition_empathy"),
        }
    except Exception:
        return {}


def set_disposition(skepticism: Optional[int] = None,
                    literality: Optional[int] = None,
                    empathy: Optional[int] = None,
                    bank: Optional[str] = None) -> bool:
    """Write disposition traits (1-5) to the bank. Best-effort."""
    try:
        updates: Dict[str, int] = {}
        if skepticism is not None:
            updates["disposition_skepticism"] = max(1, min(5, int(skepticism)))
        if literality is not None:
            updates["disposition_literalism"] = max(1, min(5, int(literality)))
        if empathy is not None:
            updates["disposition_empathy"] = max(1, min(5, int(empathy)))
        resp = requests.patch(
            _url("/config", bank),
            json={"updates": updates},
            timeout=_timeout(),
        )
        return resp.status_code == 200
    except Exception:
        return False
