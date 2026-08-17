"""Contract search service — deep retrieval seam for the unified pipeline.

One search pipeline for contract and risk search (spec:
docs/specs/unified_search_risk_merge.md):

    service = ContractSearchService(embeddings)
    rows = service.search(query, filters={"status": "completed"}, rank_by="risk")

Every returned row (ContractRow) is a Searcher-style dict whose metadata
carries identity, dates, amount, status and risk fields
(risk_score / risk_severity / matched_signals / risk_explanation) computed
unconditionally for every candidate set. Text formatting happens only at the
LLM boundary (see format_contract_results and the ticket-02 observation
formatter). gate_and_sort is never applied here — min-score filtering is a
UI-layer concern only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from apps.search._core import Searcher, _clean_text_from_enriched

from apps.search.status_labels import normalize_status as _sl_normalize_status
from apps.search.where_sql import aggregate_sql, condition_to_sql
from apps.risk_search import (
    MODE_RISKY,
    RISK_FILTER_FIELDS,
    RiskPlanner,
    normalize_filter_value,
    score_risk,
    validate_filters,
)

logger = logging.getLogger("oa-search-service")

# Regex for reference numbers like CCA20250096, CKTEST080604, etc.
_REF_NO_RE = re.compile(r"[A-Za-z]{2,}[A-Za-z0-9_-]*\d+[A-Za-z0-9_-]*")

# Semantic queries return a generous ranked set of contracts (post-dedupe).
SEMANTIC_CONTRACT_LIMIT = 50
# Chunk-level fan-out before contract-level dedupe.
SEMANTIC_CHUNK_FANOUT = SEMANTIC_CONTRACT_LIMIT * 3

# LLM observation budget: at most this many rows are written into the
# observation text; the rest is summarized by an overflow marker (the full
# set is always available to the UI via the result store).
OBSERVATION_ROW_BUDGET = 50

RANK_RELEVANCE = "relevance"
RANK_RISK = "risk"
RANK_AMOUNT = "amount"
VALID_RANKS = (RANK_RELEVANCE, RANK_RISK, RANK_AMOUNT)

# Aggregate rendering: cap on groups shown and human-readable metric titles.
MAX_AGG_GROUPS = 50
_AGG_TITLES = {
    "count": "Contract count",
    "sum_amount": "Total contract amount",
    "avg_amount": "Average contract amount",
}


def _fmt_amount(value: Any) -> str:
    """Render a numeric amount compactly, e.g. 8500000 -> HK$8.5M."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "-"
    if n >= 1_000_000:
        return "HK$%.1fM" % (n / 1_000_000)
    if n >= 1_000:
        return "HK$%.0fK" % (n / 1_000)
    return "HK$%.0f" % n


def _fmt_count(value: Any) -> str:
    """Render a count/number as a plain integer string."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "-"

# "list all"-style queries have no semantic content: they enumerate.
_ENUMERATION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:list|show|display|give\s+me|get|find)\s+(?:all|every)\b",
    re.IGNORECASE,
)
_ENUMERATION_PHRASES = {
    "all contracts", "list all", "list contracts", "show all",
    "show contracts", "list all contracts", "everything",
}


def _is_enumeration_query(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    return bool(_ENUMERATION_RE.match(text)) or text in _ENUMERATION_PHRASES


_STRUCTURED_FILTER_KEYS = (
    "contract_type", "date_from", "date_to", "counterparty_name",
    "status", "contract_id", "expired",
)


def _looks_like_ref_no(query: str) -> bool:
    return bool(query and _REF_NO_RE.fullmatch(query.strip()))


def _normalize_status(value: Any) -> Optional[str]:
    """Normalize a user-supplied status filter to a canonical DB status label.

    The user query may contain loose language like "pending approval", "done",
    "completed" that does not exactly match the DB status labels. This delegates
    to the shared status_labels.normalize_status so the semantic post-filter and
    the agent resolve aliases identically (single source of truth).
    """
    return _sl_normalize_status(value)


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _parse_tags(tags_json: Any) -> Dict[str, Any]:
    if not tags_json:
        return {}
    try:
        return json.loads(tags_json) if isinstance(tags_json, str) else dict(tags_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def _contract_key(meta: Dict[str, Any]) -> Optional[str]:
    """Stable contract-level identity used for chunk-to-contract dedupe."""
    for key in ("contract_id", "ref_no"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return None


def _meta_decoded_label(meta: Dict[str, Any], field: str) -> Optional[str]:
    """Read a yes/no/na label for `field` from a metadata dict."""
    value = meta.get(field)
    if value is not None and str(value).strip() != "":
        return normalize_filter_value(value)
    decoded = meta.get("decoded_fields")
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except Exception:
            decoded = None
    if isinstance(decoded, dict):
        pair = decoded.get(field) or {}
        return normalize_filter_value(pair.get("label"))
    return None


def _passes_risk_filters(meta: Dict[str, Any], risk_filters: List[Dict[str, str]]) -> bool:
    """Check one metadata dict against validated risk filter clauses."""
    for clause in risk_filters:
        if _meta_decoded_label(meta, clause.get("field")) != clause.get("value"):
            return False
    return True


class UnifiedQueryPlanner:
    """Shared query planner: filter extraction for every query — no mode gate.

    Wraps the risk planner's intent/filter extraction but never suppresses
    results: a non-risk plan simply yields no risk filters and a relevance
    rank hint. Shared by the regex fast-routes and the ReAct tool adapter so
    filter extraction cannot diverge between paths.
    """

    def __init__(self, risk_planner: Optional[RiskPlanner] = None, **planner_kwargs):
        self._risk_planner = risk_planner or RiskPlanner(**planner_kwargs)

    def plan(self, query: str) -> Dict[str, Any]:
        raw = self._risk_planner.plan(query)
        filters = validate_filters(raw.get("filters"))
        risk_intent = raw.get("mode") == MODE_RISKY
        return {
            "filters": filters,
            "risk_intent": risk_intent,
            "rank_hint": RANK_RISK if risk_intent else RANK_RELEVANCE,
            "explanation": str(raw.get("explanation") or ""),
        }


class ContractSearchService:
    """Deep module: one place where contract retrieval behaviour lives.

    Parameters
    ----------
    embeddings:
        A loaded txtai embeddings object. Optional if searcher is supplied.
    searcher:
        An existing Searcher instance. If omitted, one is built from
        embeddings.
    planner:
        Optional UnifiedQueryPlanner. When supplied and the caller passes no
        explicit structured filters, the planner extracts risk filters and a
        rank hint from the raw query. The planner never gates results; the
        last plan is exposed as ``last_plan`` for UI hints (e.g. auto-toggle).
    """

    def __init__(
        self,
        embeddings: Any = None,
        searcher: Optional[Searcher] = None,
        planner: Optional[UnifiedQueryPlanner] = None,
    ):
        if searcher is not None:
            self._searcher = searcher
        elif embeddings is not None:
            self._searcher = Searcher(embeddings)
        else:
            raise ValueError("ContractSearchService requires either embeddings or a searcher")
        self._planner = planner
        self.last_plan: Optional[Dict[str, Any]] = None

    # ── public seam ────────────────────────────────────────────────
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        rank_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return ContractRow dicts matching query and filters.

        Candidate semantics:
          - exact ref_no lookups: deterministic bypass (unchanged);
          - structured/exact-filter queries: ALL matching contracts, uncapped;
          - semantic queries: ~SEMANTIC_CONTRACT_LIMIT contracts after
            chunk-to-contract dedupe, ranked by relevance.
        Risk scoring is unconditional; rank_by="risk" sorts by score
        descending, "relevance" preserves retrieval order.
        """
        filters = dict(filters or {})
        q = (query or "").strip()
        self.last_plan = None

        plan_eligible = bool(q) and not _looks_like_ref_no(q)
        if (
            self._planner is not None
            and plan_eligible
            and not self._has_structured_filters(filters)
        ):
            plan = self._planner.plan(q)
            self.last_plan = plan
            plan_filters = plan.get("filters") or []
            if plan_filters:
                filters["risk_filters"] = plan_filters
            if rank_by is None:
                rank_by = plan.get("rank_hint")

        rank_by = rank_by if rank_by in VALID_RANKS else RANK_RELEVANCE

        # 1. Exact reference number lookup bypasses semantic ranking.
        if _looks_like_ref_no(q):
            exact, ref_exists = self._exact_ref_search(q, filters, limit=limit)
            if exact or ref_exists:
                # exact matches, or a ref excluded by filters (empty result)
                return self._finalize(exact, rank_by, limit)

        # 2. Structured/enumeration queries: every match, uncapped.
        if self._is_structured_query(q, filters):
            rows = self._structured_search(filters)
        else:
            # 3. Free-text: hybrid semantic search + post-filters, deduped
            #    to contract level (~SEMANTIC_CONTRACT_LIMIT contracts).
            rows = self._semantic_search(q, filters)
        return self._finalize(rows, rank_by, limit)

    # ── candidate-set builders ─────────────────────────────────────

    def _semantic_search(self, q: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = self._searcher.search(
            query=q,
            mode="hybrid",
            limit=SEMANTIC_CHUNK_FANOUT,
            expand=False,
            label_filter=filters.get("contract_type"),
            date_from=filters.get("date_from"),
            date_to=filters.get("date_to"),
            use_rerank=False,
        )
        filtered = self._apply_post_filters(chunks, filters)
        risk_filters = validate_filters(filters.get("risk_filters"))
        seen: set = set()
        rows: List[Dict[str, Any]] = []
        for r in filtered:
            meta = r.get("metadata") or {}
            if risk_filters and not _passes_risk_filters(meta, risk_filters):
                continue
            key = _contract_key(meta) or str(r.get("id"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
            if len(rows) >= SEMANTIC_CONTRACT_LIMIT:
                break
        return rows

    def _structured_search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """All contracts matching the structured filters, uncapped, one row
        per contract (first section wins as the representative snippet)."""
        risk_filters = validate_filters(filters.get("risk_filters"))
        seen: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for doc_id, text, meta in self._load_all_sections():
            key = _contract_key(meta) or str(doc_id)
            if key in seen:
                continue
            if not self._passes_filters(meta, filters):
                continue
            if risk_filters and not _passes_risk_filters(meta, risk_filters):
                continue
            seen[key] = {
                "id": doc_id,
                "text": text or "",
                "score": None,
                "metadata": dict(meta),
            }
            order.append(key)
        return [seen[k] for k in order]

    def _load_all_sections(self) -> List[Tuple[Any, str, Dict[str, Any]]]:
        """Read (id, text, metadata) for every section in the index DB."""
        db = getattr(getattr(self._searcher, "embeddings", None), "database", None)
        if db is None:
            return []
        conn = db.connection
        cur = conn.cursor()
        try:
            try:
                cur.execute("SELECT id, text, tags FROM sections")
                rows = [(r[0], r[1] or "", r[2]) for r in cur.fetchall()]
            except Exception:
                cur.execute("SELECT id, tags FROM sections")
                rows = [(r[0], "", r[1]) for r in cur.fetchall()]
        finally:
            try:
                cur.close()
            except Exception:
                pass
        return [(doc_id, text, _parse_tags(tags)) for doc_id, text, tags in rows]

    # ── contracts_where: exact structured retrieval (ticket 05) ────

    def search_where(
        self,
        condition: str,
        *,
        limit: Optional[int] = None,
        llm_client: Any = None,
        allow_llm: bool = True,
        rank_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Exact structured retrieval: natural-language condition -> SQL.

        Translation chain (ported from corpchat search_messages_where):
        rule-based first, LLM text-to-SQL as validated fallback. Bare
        "list all"-style conditions have no filter content -> no WHERE
        clause -> every contract. When translation fails entirely, falls
        back to a semantic index scan over the same corpus; this method
        never raises to the caller. Rows are the same ContractRow shape
        as search() (chunk dedupe + unconditional risk scoring).
        """
        cond = (condition or "").strip()
        self.last_plan = None
        rank_by = rank_by if rank_by in VALID_RANKS else RANK_RELEVANCE

        sql = None
        if cond:
            sql = condition_to_sql(cond, llm_client=llm_client, allow_llm=allow_llm)
        else:
            sql = condition_to_sql("", allow_llm=False)  # no-WHERE: all contracts

        if sql is None:
            # Untranslatable condition -> index-scan fallback (semantic).
            logger.info("contracts_where fallback to semantic scan: %r", cond)
            return self._finalize(self._semantic_search(cond, {}), rank_by, limit)

        section_rows = self._run_sections_sql(sql)
        if section_rows is None:
            # SQL execution failed -> index-scan fallback; never raise.
            logger.warning("contracts_where SQL failed; semantic fallback: %r", cond)
            if cond:
                return self._finalize(self._semantic_search(cond, {}), rank_by, limit)
            return []

        seen: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for doc_id, text, meta in section_rows:
            key = _contract_key(meta) or str(doc_id)
            if key in seen:
                continue
            seen[key] = {
                "id": doc_id,
                "text": text or "",
                "score": None,
                "metadata": dict(meta),
            }
            order.append(key)
        rows = [seen[k] for k in order]
        return self._finalize(rows, rank_by, limit)

    def aggregate(
        self,
        metric: str,
        group_by: str = "",
        condition: str = "",
    ) -> str:
        """SQL-side aggregate over the sections table, rendered as a text table.

        The database computes the aggregate over the FULL matching set (never a
        LIMIT-capped row fetch), so totals are correct. metric/group_by are
        whitelisted in where_sql.aggregate_sql; unknown values or a missing DB
        yield a human-readable message (never an exception to the agent).
        """
        sql = aggregate_sql(metric, group_by, condition)
        if sql is None:
            return ("Unsupported aggregate: metric=%r group_by=%r. Supported "
                    "metrics: count, sum_amount, avg_amount; groups: department, "
                    "counterparty_name, contract_type, status_label, year."
                    % (metric, group_by))

        db = getattr(getattr(self._searcher, "embeddings", None), "database", None)
        if db is None:
            return "Aggregate unavailable: no sections database connection."
        conn = db.connection
        try:
            cur = conn.cursor()
            cur.execute(sql)
            fetched = cur.fetchall()
        except Exception as e:
            logger.warning("aggregate SQL failed (%s): %s", e, sql)
            return "Aggregate query failed; please rephrase."

        is_amount = (metric or "").strip().lower() in ("sum_amount", "avg_amount")
        label = _AGG_TITLES.get((metric or "").strip().lower(), "Value")

        if not group_by:
            # Single overall figure (no GROUP BY): one row, value in col 0.
            value = fetched[0][0] if fetched and fetched[0] else None
            rendered = _fmt_amount(value) if is_amount else _fmt_count(value)
            return "%s overall: %s" % (label, rendered)

        groups = [(str(k) if k not in (None, "") else "(unknown)", v)
                  for k, v in fetched]
        if not groups:
            return "No matching contracts."

        total_groups = len(groups)
        shown = groups[:MAX_AGG_GROUPS]
        overflow = total_groups - len(shown)

        key_head = group_by or "group"
        val_head = {"count": "count", "sum_amount": "total_amount",
                    "avg_amount": "avg_amount"}.get(metric, "value")
        width = max([len(key_head)] + [len(k) for k, _ in shown])
        lines = ["%s by %s (%d groups):" % (label, group_by, total_groups),
                 "%s | %s" % (key_head.ljust(width), val_head),
                 "%s-+-%s" % ("-" * width, "-" * len(val_head))]
        for k, v in shown:
            rendered = _fmt_amount(v) if is_amount else _fmt_count(v)
            lines.append("%s | %s" % (k.ljust(width), rendered))
        if overflow > 0:
            lines.append("+%d more groups not shown" % overflow)
        return "\n".join(lines)

    def _run_sections_sql(self, sql: str) -> Optional[List[Tuple[Any, str, Dict[str, Any]]]]:
        """Execute a validated read-only SELECT on the sections table.

        Returns (id, text, metadata) triples, or None on any failure so
        callers can fall back to an index scan.
        """
        db = getattr(getattr(self._searcher, "embeddings", None), "database", None)
        if db is None:
            return None
        conn = db.connection
        cur = conn.cursor()
        try:
            cur.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            if "id" not in cols or "tags" not in cols:
                return None
            out: List[Tuple[Any, str, Dict[str, Any]]] = []
            for r in cur.fetchall():
                rec = dict(zip(cols, r))
                out.append((rec["id"], rec.get("text") or "", _parse_tags(rec.get("tags"))))
            return out
        except Exception as e:
            logger.warning("sections SQL failed (%s): %s", e, sql)
            return None
        finally:
            try:
                cur.close()
            except Exception:
                pass

    # ── risk scoring + ranking (unconditional, never gating) ───────

    def _finalize(
        self,
        rows: List[Dict[str, Any]],
        rank_by: str,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        self._score_rows(rows)
        if rank_by == RANK_RISK:
            rows = sorted(
                rows,
                key=lambda r: -((r.get("metadata") or {}).get("risk_score") or 0),
            )
        elif rank_by == RANK_AMOUNT:
            rows = sorted(
                rows,
                key=lambda r: -((r.get("metadata") or {}).get("amount") or 0),
            )
        if limit is not None:
            rows = rows[:limit]
        return rows

    def _score_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Attach risk_score/severity/signals/explanation to every row in place."""
        if not rows:
            return
        records: List[Dict[str, Any]] = []
        for r in rows:
            meta = r.get("metadata") or {}
            rec: Dict[str, Any] = {}
            decoded = meta.get("decoded_fields")
            if isinstance(decoded, str):
                try:
                    decoded = json.loads(decoded)
                except Exception:
                    decoded = None
            if isinstance(decoded, dict):
                for field, pair in decoded.items():
                    rec[field] = (pair or {}).get("label")
                rec["decoded_fields"] = decoded
            for field in RISK_FILTER_FIELDS:
                if meta.get(field) not in (None, ""):
                    rec[field] = meta.get(field)
            records.append(rec)
        scored = score_risk(pd.DataFrame(records))
        for r, (_, srow) in zip(rows, scored.iterrows()):
            meta = r.setdefault("metadata", {})
            score = int(srow.get("risk_score") or 0)
            severity = str(srow.get("risk_severity") or "low")
            signals = list(srow.get("matched_signals") or [])
            explanation = str(srow.get("risk_explanation") or "")
            meta["risk_score"] = score
            meta["risk_severity"] = severity
            meta["matched_signals"] = signals
            meta["risk_explanation"] = explanation
            r["risk_score"] = score
            r["risk_severity"] = severity
            r["matched_signals"] = signals
            r["risk_explanation"] = explanation

    # ── filter helpers ─────────────────────────────────────────────

    @staticmethod
    def _is_structured_query(q: str, filters: Dict[str, Any]) -> bool:
        """Structured path = risk filter clauses, "list all"-style
        enumeration, or an empty query with filters. Free-text queries with
        filters stay semantic (retrieval quality) with post-filtering."""
        if validate_filters(filters.get("risk_filters")):
            return True
        if _is_enumeration_query(q):
            return True
        return not q.strip()

    @staticmethod
    def _has_structured_filters(filters: Dict[str, Any]) -> bool:
        if any(filters.get(k) not in (None, "") for k in _STRUCTURED_FILTER_KEYS):
            return True
        return bool(validate_filters(filters.get("risk_filters")))

    def _exact_ref_search(
        self,
        ref_no: str,
        filters: Dict[str, Any],
        *,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Deterministic exact ref_no lookup against the sections DB.

        Returns (rows, ref_exists). ref_exists is True when at least one
        section has the requested ref_no regardless of filter compatibility,
        so callers can avoid an unnecessary semantic fallback for a
        filtered-out exact match.
        """
        db = getattr(getattr(self._searcher, "embeddings", None), "database", None)
        if db is None:
            return [], False

        conn = db.connection
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, tags FROM sections")
            rows = cur.fetchall()
        finally:
            try:
                cur.close()
            except Exception:
                pass

        wanted = ref_no.strip().upper()
        out: List[Dict[str, Any]] = []
        ref_exists = False
        risk_filters = validate_filters(filters.get("risk_filters"))
        for row in rows:
            doc_id, tags_json = (row[0], row[-1]) if len(row) >= 2 else (None, None)
            if doc_id is None:
                continue
            meta = _parse_tags(tags_json)
            if str(meta.get("ref_no") or "").strip().upper() != wanted:
                continue
            ref_exists = True
            if not self._passes_filters(meta, filters):
                continue
            if risk_filters and not _passes_risk_filters(meta, risk_filters):
                continue
            doc = self._searcher._fetch_one_doc(doc_id)
            if doc:
                out.append(doc)
            if limit is not None and len(out) >= limit:
                break
        return out, ref_exists

    def _apply_post_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter semantic results by structured fields not handled by txtai."""
        party = (filters.get("counterparty_name") or "").strip().lower()
        status = _normalize_status(filters.get("status"))
        contract_id = (filters.get("contract_id") or "").strip().lower()
        expired = _parse_bool(filters.get("expired"))

        out = []
        for r in results:
            meta = r.get("metadata") or {}
            if party and party not in str(meta.get("counterparty_name") or "").lower():
                continue
            if status:
                actual = _normalize_status(meta.get("status_label") or meta.get("status") or "")
                if actual != status:
                    continue
            if contract_id:
                refs = [str(meta.get(k) or "").lower() for k in ("ref_no", "contract_id", "id")]
                if not any(contract_id in ref for ref in refs):
                    continue
            if expired is not None:
                if _parse_bool(meta.get("expired")) is not expired:
                    continue
            out.append(r)
        return out

    def _passes_filters(self, meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check whether a single metadata dict satisfies all filters."""
        ctype = filters.get("contract_type")
        if ctype:
            wanted = {p.strip() for p in str(ctype).split(",") if p.strip()}
            actual = str(meta.get("contract_type") or "")
            actual_label = str(meta.get("contract_type_label") or "")
            if actual not in wanted and actual_label.lower() not in {w.lower() for w in wanted}:
                return False

        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        date_value = str(meta.get("requested_date") or "")
        if date_from and date_value and date_value < str(date_from):
            return False
        if date_to and date_value and date_value > str(date_to):
            return False

        filtered = self._apply_post_filters([{"metadata": meta}], filters)
        return bool(filtered)


def format_contract_observation(
    rows: List[Dict[str, Any]],
    *,
    budget: int = OBSERVATION_ROW_BUDGET,
) -> str:
    """Format ContractRow dicts into the LLM observation (ticket 02).

    Richer than the legacy formatter: each row carries ref, counterparty,
    type, status, start/end dates, amount, risk score/severity, top signals,
    and a ~200-char snippet. At most  rows are written; overflow is
    a single marker line. Empty input -> empty string (the agent turns that
    into "No matching contracts were found.").
    """
    if not rows:
        return ""
    out = []
    for i, r in enumerate(rows[:budget], 1):
        meta = r.get("metadata", {})
        ref = meta.get("ref_no") or "?"
        party = meta.get("counterparty_name") or meta.get("title") or "?"
        ctype = meta.get("contract_type_label") or meta.get("contract_type", "-")
        status = meta.get("status_label") or meta.get("status") or "-"
        start = meta.get("contract_start_date") or meta.get("requested_date") or "-"
        end = meta.get("contract_end_date") or "-"
        amount = meta.get("amount_label") or meta.get("amount") or "-"
        score = meta.get("risk_score", r.get("risk_score", 0)) or 0
        severity = meta.get("risk_severity") or r.get("risk_severity") or "low"
        signals = meta.get("matched_signals") or r.get("matched_signals") or []
        signals_str = "; ".join(str(s2) for s2 in signals[:3]) if signals else "none"
        snippet = _clean_text_from_enriched(r.get("text", ""))[:200]
        out.append(
            "%d. [ref=%s | %s | type=%s | status=%s | %s -> %s | amount=%s] "
            "risk=%s (%s); signals: %s; %s"
            % (i, ref, party, ctype, status, start, end, amount,
               score, severity, signals_str, snippet)
        )
    overflow = len(rows) - budget
    if overflow > 0:
        out.append("(+%d more contracts not shown - see results table)" % overflow)
    return chr(10).join(out)


def format_contract_results(results: List[Dict[str, Any]]) -> str:
    """Format Searcher-style result rows into an LLM-readable observation.

    NOTE: superseded by the ticket-02 observation formatter (richer fields,
    50-row budget, overflow marker); kept for backward compatibility.
    """
    if not results:
        return ""
    out = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        text = _clean_text_from_enriched(r.get("text", ""))[:200]
        ref = meta.get("ref_no") or "?"
        party = meta.get("counterparty_name") or meta.get("title") or "?"
        ctype = meta.get("contract_type_label") or meta.get("contract_type", "-")
        out.append(f"{i}. [ref={ref} | {party} | type={ctype}] {text}")
    return "\n".join(out)
