#!/usr/bin/env python3
"""OA Contract Screening — Streamlit app."""

import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

from core.config import ensure_hf_offline

ensure_hf_offline()  # set HF_HUB_OFFLINE from cache state when unset (docker first boot)

from apps.search import (
    DEFAULT_INDEX_PATH,
    TOOL_CONTRACT_SEARCH,
    TOOL_RISK_SEARCH,
    TOOL_NONE,
    ContractSearchService,
    CrossTableAgent,
    LangChainAgent,
    Searcher,
    load_index,
)
from apps.search.result_store import clear_results, snapshot_results
from apps.search_cli import build_contract_tool, build_where_tool
try:
    from apps.search.langchain_agent import check_llm_health as _check_llm_health_uncached
except Exception:  # pragma: no cover - defensive: never crash the UI on import
    def _check_llm_health_uncached():
        return {"ok": False, "provider": "none", "model": "", "error": "langchain_agent import failed"}


@st.cache_data(ttl=300, show_spinner=False)
def check_llm_health(deep: bool = False) -> Dict[str, Any]:
    """Cached wrapper: Streamlit reruns the whole script on every widget
    interaction. Shallow mode (GET /models, ~0.1s) is enough per rerun; the
    Phase-2 chat probe (deep=True, a full remote round-trip that can take
    10s+ on a loaded proxy) only runs via the sidebar "Recheck LLM" button."""
    return _check_llm_health_uncached(deep=deep)
# Fallback clarification prompt when the agent cannot route a query.
DEFAULT_CLARIFICATION = "Could you narrow down which contract or counterparty you mean?"
from apps.attachment_summary import (attachment_label, human_file_size, list_attachments,
    resolve_attachment_path, summarize_contract_with_attachment)
from apps.detail_view import build_contextual_groups, coalesce_raw, humanize_signals
from apps.risk_search import score_risk
from apps.search._core import _clean_text_from_enriched


@st.cache_resource
def _load_embeddings(index_path: str):
    return load_index(index_path)


# ─────────────────────────────────────────────────────────────────────
# Agentic layer (Epic: unified LangChain agentic UI)
# ─────────────────────────────────────────────────────────────────────
def _build_agent(index_path: str, embeddings):
    """Build the LangChain tool-calling agent over the existing pipelines.

    Falls back to the manual-ReAct CrossTableAgent only when the provider is
    genuinely unavailable or misconfigured. If the provider package is present,
    we prefer the LangChain agent.
    """
    # Unified pipeline (tickets 01-02): ONE search tool for contract and risk
    # search; the risk tool is retired as a separate LLM tool. Ticket 05 adds
    # contracts_where for exact structured retrieval (Phase 2).
    contract_tool = build_contract_tool(embeddings)
    where_tool = build_where_tool(embeddings)
    try:
        from apps.search.langchain_agent import build_default_llm, AgentConfigError, check_llm_health
    except Exception as e:  # import failure → manual agent
        st.caption(f"LangChain agent unavailable (import failed: {e}); using built-in router agent.")
        return CrossTableAgent(contract_tool=contract_tool, where_tool=where_tool)

    try:
        llm = build_default_llm(api_key=os.getenv("LITELLM_API_KEY", ""))
    except AgentConfigError as e:
        st.caption(f"LangChain agent unavailable ({e}); using built-in router agent.")
        return CrossTableAgent(contract_tool=contract_tool, where_tool=where_tool)

    return LangChainAgent(contract_tool=contract_tool, where_tool=where_tool, llm=llm)


def _get_summary_llm():
    """LLM for per-contract summaries; same single LiteLLM provider as the agent."""
    llm = st.session_state.get("summary_llm")
    if llm is not None:
        return llm
    try:
        from apps.search.langchain_agent import build_default_llm
        llm = build_default_llm(api_key=os.getenv("LITELLM_API_KEY", ""))
    except Exception:
        llm = None
    st.session_state["summary_llm"] = llm
    return llm


# Process-wide contract summary cache: backend keys are "<ref_no>|<attachment mtime>".
_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}


def _summary_cache_get(ref_no: str) -> Optional[Dict[str, Any]]:
    prefix = str(ref_no) + "|"
    for key, value in _SUMMARY_CACHE.items():
        if key.startswith(prefix):
            return value
    return None


def _summary_cache_drop(ref_no: str) -> None:
    prefix = str(ref_no) + "|"
    for key in [k for k in _SUMMARY_CACHE if k.startswith(prefix)]:
        del _SUMMARY_CACHE[key]


def _render_summary_result(result: Dict[str, Any]) -> None:
    if result.get("notice"):
        st.info(result["notice"])
    att = result.get("attachment")
    if att:
        st.caption("Based on: **%s** - %s"
                   % (attachment_label(att.get("field_name")),
                      att.get("file_name") or "(unnamed file)"))
    st.markdown(result.get("summary") or "_(no summary)_")


def _generate_contract_summary(row: Any, ref_no: str) -> None:
    with st.spinner("Generating summary (this calls the LLM)..."):
        try:
            result = summarize_contract_with_attachment(
                row, llm=_get_summary_llm(), cache=_SUMMARY_CACHE)
        except Exception as exc:
            logger.warning("Summary generation failed for %s: %s", ref_no, exc)
            result = None
    if result is None:
        st.warning("Summary generation failed; see logs.")
    else:
        _render_summary_result(result)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_contract_search(index_path: str, query: str, filters_json: str) -> List[Dict[str, Any]]:
    """Cached evidence search keyed on primitives (corpchat _run_search pattern).

    Avoids re-running the full txtai+sqlite search on every Streamlit rerun
    (e.g. when the user switches the contract-detail selectbox).
    """
    embeddings = _load_embeddings(index_path)
    service = ContractSearchService(searcher=Searcher(embeddings))
    return service.search(query, filters=json.loads(filters_json or "{}"), limit=10)


def _run_contract_search(searcher: Searcher, query: str, filters: Dict[str, Any]):
    """Re-run the underlying contract search to retrieve evidence rows."""
    index_path = st.session_state.get("index_path") or DEFAULT_INDEX_PATH
    filters_json = json.dumps(filters or {}, sort_keys=True, default=str)
    return _cached_contract_search(index_path, query, filters_json)


def _render_agent_metadata(result: Dict[str, Any]):
    """Surface the agentic workflow with compact metadata."""
    intent = result.get("intent", "-")
    tool = result.get("tool", TOOL_NONE)
    fallback = result.get("fallback", False)
    tool_label = {
                TOOL_CONTRACT_SEARCH: "Contract search (normal)",
        TOOL_RISK_SEARCH: "Risk search (flagged contracts)",
    }.get(tool, "None — clarification")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Intent: **{intent}**")
    with col2:
        st.caption(f"Tool: **{tool_label}**")
    with col3:
        st.caption(f"Routing: **{'Fallback' if fallback else 'LangChain'}**")

    with st.expander("Agent steps"):
        for step in result.get("steps", []):
            st.markdown(f"{step.get('icon', '')} **{step.get('label', '')}** — {step.get('detail', '')}")
        for call in result.get("tool_calls", []):
            tool = call.get("tool", "-")
            tool_input = call.get("tool_input", "")
            filters = call.get("filters") or {}
            st.markdown(f"- **{tool}** · input `{tool_input}` · filters `{filters}`")
    if fallback:
        st.warning(
            "⚠️ Large Language Model (LLM) is unreachable or too slow. "
            "Showing deterministic keyword-based results. Searches may miss nuance "
            "until the LLM proxy recovers."
        )
@st.cache_data(ttl=300)
def _load_sections(index_path: str) -> pd.DataFrame:
    db_path = os.path.join(index_path, "documents")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT id, tags FROM sections", conn)
    finally:
        conn.close()

    # First pass: collect every decoded + contextual field that actually appears
    # in the data (drives "display all columns" without showing always-NULL ones).
    parsed: List[Dict[str, Any]] = []
    decoded_keys: set = set()
    contextual_keys: set = set()
    for _, row in df.iterrows():
        try:
            meta = json.loads(row["tags"]) if row["tags"] else {}
        except Exception:
            meta = {}
        decoded = meta.get("decoded_fields") or {}
        contextual = meta.get("contextual_fields") or {}
        parsed.append((row, meta, decoded, contextual))
        decoded_keys.update(decoded.keys())
        contextual_keys.update(contextual.keys())

    decoded_cols = sorted(decoded_keys)
    contextual_cols = sorted(contextual_keys)

    records: List[Dict[str, Any]] = []
    for row, meta, decoded, contextual in parsed:
        rec: Dict[str, Any] = {
            "id": row["id"],
            "contract_id": meta.get("contract_id"),
            "ref_no": meta.get("ref_no"),
            "title": meta.get("title"),
            "counterparty_name": meta.get("counterparty_name"),
            "department": meta.get("department"),
            "amount": meta.get("amount"),
            "amount_label": meta.get("amount_label"),
            "contract_start_date": meta.get("contract_start_date"),
            "contract_end_date": meta.get("contract_end_date"),
            "requested_date": meta.get("requested_date"),
            "status": meta.get("status_label") or meta.get("status"),
            "contract_type": meta.get("contract_type"),
            "contract_type_label": meta.get("contract_type_label"),
            "chunk_index": meta.get("chunk_index"),
        }
        # Flatten every decoded label present in the data (e.g. Over5M -> "yes").
        for field in decoded_cols:
            rec[field] = (decoded.get(field) or {}).get("label")
        # Flatten every contextual field present (verbatim raw values).
        for field in contextual_cols:
            rec[field] = contextual.get(field)
        # Keep the structured dicts for detail views.
        rec["decoded_fields"] = decoded
        rec["contextual_fields"] = contextual
        rec["raw"] = meta.get("raw")  # full source record (post-rebuild indexes)
        records.append(rec)
    return pd.DataFrame(records)


def _options(df: pd.DataFrame, column: str) -> List[str]:
    return sorted({str(v) for v in df[column].dropna().tolist() if str(v).strip() != ""})



def _filter_contracts(
    df: pd.DataFrame,
    query: str,
    contract_types: Optional[List[str]] = None,
    departments: Optional[List[str]] = None,
    counterparties: Optional[List[str]] = None,
    statuses: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if query:
        q = query.lower()
        text_cols = ["title", "counterparty_name", "department", "ref_no"]
        mask = False
        for col in text_cols:
            mask = mask | out[col].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        out = out[mask]
    if contract_types:
        out = out[out["contract_type"].astype(str).isin(contract_types)]
    if departments:
        out = out[out["department"].astype(str).isin(departments)]
    if counterparties:
        out = out[out["counterparty_name"].astype(str).isin(counterparties)]
    if statuses:
        out = out[out["status"].astype(str).isin(statuses)]
    return out


def _decoded_label(meta: Dict[str, Any], field: str) -> Optional[str]:
    pair = (meta.get("decoded_fields") or {}).get(field) or {}
    return pair.get("label")


def _summarize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        meta = r.get("metadata", {})
        rows.append({
            "id": r.get("id"),
            "score": r.get("score"),
            "counterparty_name": meta.get("counterparty_name"),
            "department": meta.get("department"),
            "contract_type": meta.get("contract_type_label") or meta.get("contract_type"),
            "requested_date": meta.get("requested_date"),
            "status": meta.get("status_label") or meta.get("status"),
            "amount_label": meta.get("amount_label"),
            # decoded boolean/risk labels (raw-first)
            "Over5M": _decoded_label(meta, "Over5M"),
            "Over100M": _decoded_label(meta, "Over100M"),
            "FlagNeedLegal": _decoded_label(meta, "FlagNeedLegal"),
            "IsRisksAccepted": _decoded_label(meta, "IsRisksAccepted"),
            "PreliminaryReviewFlag": _decoded_label(meta, "PreliminaryReviewFlag"),
            "IfRelatedToData": _decoded_label(meta, "IfRelatedToData"),
            "title": meta.get("title"),
            "ref_no": meta.get("ref_no"),
        })
    return pd.DataFrame(rows)


def _render_agentic(index_path: str):
    """Unified agentic contract search UI.

    One natural-language box. The agent classifies the query, routes to the
    unified contract search tool, surfaces intent/tool/fallback metadata,
    asks a clarifying question when needed, and shows supporting evidence.
    """
    st.session_state["index_path"] = index_path  # for cached evidence search
    st.subheader("Ask")
    st.caption(
        "Ask in natural language — e.g. 'which contracts mention unlimited liability', "
        "'show completed contracts with Acme', or 'is CCA20250096 expired'. The agent "
        "maps filters and searches the contract corpus automatically."
    )

    embeddings = _load_embeddings(index_path)
    searcher = Searcher(embeddings)
    agent = _build_agent(index_path, embeddings)

    query = st.text_input(
        "Your question",
        value="show contracts where risk was not accepted",
        key="agentic_query",
    )

    if st.button("Ask", type="primary"):
        progress = st.empty()
        def _stage(icon, msg):
            progress.markdown(f"**{icon}** {msg}")
        # Keep the current query in a non-widget session key for reruns / evidence lookups.
        st.session_state["agentic_last_query"] = query
        clear_results()  # result store: drop the previous turn's stash
        with st.spinner("Thinking..."):
            try:
                result = agent.process(query, on_stage=_stage)
            except TypeError as e:
                if "unexpected keyword argument 'on_stage'" in str(e):
                    result = agent.process(query)
                else:
                    raise
        progress.empty()
        st.session_state["agentic_result"] = result

    result = st.session_state.get("agentic_result")
    if result is None:
        return

    _render_agent_metadata(result)

    if not result.get("success") and result.get("tool") == TOOL_NONE:
        st.warning(result.get("output") or DEFAULT_CLARIFICATION)
        return

    if result.get("clarify"):
        st.warning(result.get("output") or DEFAULT_CLARIFICATION)
        return

    # All queries (including risk intent) render via the unified contract pipeline.
    _render_agentic_contract(result, searcher)


# ── Ask UI merged table (ticket 03) ──────────────────────────────
# Default min risk score for the table filter (calibrated: >=80 keeps ~8 of
# ~260 contracts; UI-only, the pipeline itself never gates).
DEFAULT_MIN_RISK_SCORE = 80

_TABLE_BASE_COLUMNS = [
    "ref_no", "counterparty_name", "contract_type", "status",
    "amount_label", "risk_score", "risk_severity",
]
_TABLE_EXTRA_COLUMNS = [
    "department", "requested_date", "expired", "matched_signals",
    "risk_explanation", "snippet",
]


def _flatten_result_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten service ContractRow dicts into display records for the table."""
    flat: List[Dict[str, Any]] = []
    for r in rows:
        meta = r.get("metadata") or {}
        signals = meta.get("matched_signals") or r.get("matched_signals") or []
        flat.append({
            "ref_no": meta.get("ref_no") or r.get("ref_no"),
            "title": meta.get("title"),
            "counterparty_name": meta.get("counterparty_name"),
            "contract_type": meta.get("contract_type_label") or meta.get("contract_type"),
            "status": meta.get("status_label") or meta.get("status"),
            "contract_start_date": meta.get("contract_start_date") or meta.get("requested_date"),
            "contract_end_date": meta.get("contract_end_date"),
            "amount_label": meta.get("amount_label") or meta.get("amount"),
            "risk_score": meta.get("risk_score", r.get("risk_score", 0)) or 0,
            "risk_severity": meta.get("risk_severity") or r.get("risk_severity") or "low",
            "department": meta.get("department"),
            "requested_date": meta.get("requested_date"),
            "expired": meta.get("expired"),
            "matched_signals": "; ".join(str(x) for x in signals),
            "risk_explanation": meta.get("risk_explanation") or r.get("risk_explanation"),
            "snippet": _clean_text_from_enriched(r.get("text") or "")[:200],
        })
    return flat


def _apply_table_controls(
    flat: List[Dict[str, Any]],
    *,
    rank_by_risk: bool,
    min_score: Optional[int],
    columns: Optional[List[str]],
) -> pd.DataFrame:
    """Apply UI-only controls: min-score filter, risk ranking, column pick.

    This never changes the underlying result set - the store keeps all rows.
    """
    threshold = min_score if min_score is not None else 0
    shown = [r for r in flat if (r.get("risk_score") or 0) >= threshold]
    if rank_by_risk:
        shown = sorted(shown, key=lambda r: -(r.get("risk_score") or 0))
    valid = _TABLE_BASE_COLUMNS + _TABLE_EXTRA_COLUMNS
    cols = [c for c in (columns or []) if c in valid] or list(_TABLE_BASE_COLUMNS)
    if not shown:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(shown)[cols]


def _render_agentic_contract(result: Dict[str, Any], searcher: Searcher):
    """Unified evidence table + RAG answer, rendered from the result store.

    The unified search tool stashes its full ContractRow set per invocation
    (the store is cleared before each agent turn), so the UI renders the
    structured rows directly - no observation-text parsing, no re-query.
    Falls back to a re-query only when the store holds no rows for this turn
    (e.g. tool error or a legacy agent path).
    """
    tool_call = (result.get("tool_calls") or [{}])[0]
    t_query = st.session_state.get("agentic_last_query") or tool_call.get("tool_input") or ""
    filters = tool_call.get("filters") or {}

    snap = snapshot_results()
    rows = snap.get("rows") or []
    if not rows:
        _render_contract_evidence(result, searcher, t_query, filters)
        return

    # 1. Answer generation (RAG answer) -- shown first
    if result.get("output"):
        with st.expander("Answer generation", expanded=True):
            st.markdown("**LLM summary**")
            st.markdown(result.get("output") or "(no summary returned)")
            st.caption("This summary is generated from the retrieved contract evidence.")

    # 2. Merged evidence table with UI-only controls (ticket 03):
    #    rank-by-risk toggle (default from the planner rank hint), min-score
    #    filter (default 80), column picker. The store keeps the full set.
    key_suffix = re.sub(r"[^0-9A-Za-z]+", "_", t_query or "results")[-40:]
    flat = _flatten_result_rows(rows)
    c1, c2 = st.columns([1, 3])
    with c1:
        rank_by_risk = st.toggle(
            "Rank by risk",
            value=(snap.get("rank_by") == "risk"),
            key="rank_risk_" + key_suffix,
        )
        last_score_key = "last_min_score_" + key_suffix
        if last_score_key not in st.session_state:
            st.session_state[last_score_key] = DEFAULT_MIN_RISK_SCORE
        if rank_by_risk:
            st.session_state[last_score_key] = st.number_input(
                "Min risk score", min_value=0, max_value=1000,
                value=st.session_state[last_score_key], step=10,
                key="min_score_" + key_suffix,
            )
        min_score = st.session_state[last_score_key] if rank_by_risk else 0
    table = _apply_table_controls(
        flat, rank_by_risk=rank_by_risk, min_score=min_score,
        columns=st.session_state.get("sidebar_columns_", _TABLE_BASE_COLUMNS))
    st.success(f"{len(table)} of {len(flat)} supporting contract(s) shown")
    st.dataframe(table, use_container_width=True)
    obs_count = snap.get("observation_count")
    if obs_count is not None and snap.get("total", 0) > obs_count:
        st.caption(
            f"The answer summary covers the first {obs_count} of "
            f"{snap['total']} contracts; the table always shows the full set."
        )

    # 3. Per-contract detail selectbox (merged view, keyed by ref_no)
    refs = [r for r in table.get("ref_no", []) if r] if not table.empty else []
    refs = list(dict.fromkeys(refs))
    if refs:
        st.markdown("---")
        selected = st.selectbox("View detail", refs, key="detail_select")
        if selected:
            _render_contract_detail(selected, getattr(searcher, "index_path", None)
                                    or st.session_state.get("index_path", ""))


def _render_contract_detail(ref_no: str, index_path: str):
    """Merged per-contract detail view, shared by contract search, risk search,
    and the Browse tab. Data comes from the cached sections dataframe (keyed by
    ref_no); risk is always computed on the fly via score_risk (deterministic,
    no LLM); attachments are listed from the contract_attachments table.
    """
    st.caption("Contract: " + str(ref_no))
    try:
        df = _load_sections(index_path)
    except Exception:
        df = None  # index store unavailable -> degrade, never crash the view
    if df is None:
        st.info("Record lookup unavailable (index store not loaded).")
        return
    if "ref_no" in df.columns:
        match = df[df["ref_no"].astype(str) == str(ref_no)]
    else:
        match = df.iloc[0:0]
    if match.empty:
        st.info("No record found for " + str(ref_no))
        return
    row = match.iloc[0]  # contract-level fields repeat across chunks; first row suffices

    tab_summary, tab_raw, tab_contextual, tab_risk, tab_attachments = st.tabs(
        ["Summary", "Raw", "Contextual", "Risk", "Attachments"]
    )
    with tab_summary:
        st.caption("Generate an English review summary: the recorded risk assessment "
                   "plus a summary of the signed contract attachment. "
                   "Generation uses the configured LLM.")
        ref_key = str(ref_no)
        if _summary_cache_get(ref_key) is not None:
            _render_summary_result(_summary_cache_get(ref_key))
            if st.button("Regenerate", key="regen_" + ref_key):
                _summary_cache_drop(ref_key)
                _generate_contract_summary(row, ref_key)
        else:
            if st.button("Generate summary", key="gen_" + ref_key):
                _generate_contract_summary(row, ref_key)
    with tab_raw:
        # Raw lens: the complete stored record as JSON, no curation. Once the
        # index metadata carries the raw record (next rebuild), this is the full
        # ~155-column source record; until then it is every indexed field.
        record, is_full_raw = coalesce_raw(row.get("raw"), row.to_dict())
        if not is_full_raw:
            st.caption("Showing the complete stored index record; the full "
                       "source record appears here after the next index rebuild.")
        st.json(json.loads(json.dumps(record, default=str)))
    with tab_contextual:
        # Contextual lens: curated self-explanatory fields + decoded labels,
        # grouped; missing values render an explicit placeholder.
        for group in build_contextual_groups(row.to_dict()):
            st.markdown("**" + group["group"] + "**")
            for f in group["fields"]:
                st.markdown("- " + f["label"] + ": " + str(f["value"]))
    with tab_risk:
        try:
            scored = score_risk(match.head(1))
            r = scored.iloc[0]
            signals = humanize_signals(r.get("matched_signals") or [])
            st.markdown("Risk score: **%s** | Severity: **%s**"
                        % (r.get("risk_score", 0), r.get("risk_severity", "-")))
            if signals:
                st.markdown("**Signals:**")
                for sig in signals:
                    st.markdown("- " + str(sig))
            else:
                st.info("No risk flags detected.")
        except Exception as e:
            st.warning("Risk scoring unavailable: %s" % e)
    with tab_attachments:
        try:
            att_rows = list_attachments(row.get("contract_id"), str(ref_no))
        except Exception:
            att_rows = None
        if att_rows is None:
            st.info("Attachments unavailable.")
        elif not att_rows:
            st.info("No attachments on record.")
        else:
            for a in att_rows:
                name = a.get("file_name") or "(unnamed file)"
                line = "- **" + attachment_label(a.get("field_name")) + "** - " + str(name)
                size = human_file_size(a.get("file_size"))
                if size:
                    line += " (" + size + ")"
                st.markdown(line)


def _render_contract_evidence(result: Dict[str, Any], searcher: Searcher, t_query: str, filters: Dict[str, Any]):
    """RAG answer above, table below, per-contract selectbox at bottom."""

    # 1. Answer generation (RAG answer) -- shown first
    if result.get("output"):
        with st.expander("Answer generation", expanded=True):
            st.markdown("**LLM summary**")
            st.markdown(result.get("output") or "(no summary returned)")
            st.caption("This summary is generated from the retrieved contract evidence.")

    # 2. Evidence table
    with st.spinner("Loading supporting evidence..."):
        try:
            evidence = _run_contract_search(searcher, t_query, filters)
        except Exception as e:
            st.info(f"Could not load evidence rows: {e}")
            return

    evidence_df = evidence if isinstance(evidence, pd.DataFrame) else None
    if evidence_df is not None:
        if evidence_df.empty:
            st.info("No matching contracts found.")
            return
        rows = evidence_df.to_dict("records")
    else:
        if not evidence:
            st.info("No matching contracts found.")
            return
        rows = list(evidence)

    st.success(f"{len(rows)} supporting contract(s)")
    st.dataframe(_summarize_results(rows), use_container_width=True)

    # 3. Per-contract detail selectbox (merged view, keyed by ref_no)
    refs = list(dict.fromkeys(
        ref for ref in (
            (r.get("metadata", {}) or {}).get("ref_no") or r.get("ref_no")
            for r in rows
        ) if ref
    ))
    if refs:
        st.markdown("---")
        selected = st.selectbox("View detail", refs, key="detail_select")
        if selected:
            _render_contract_detail(selected, getattr(searcher, "index_path", None)
                                    or st.session_state.get("index_path", ""))


def _render_browser(index_path: str):
    st.subheader("Browse")
    df = _load_sections(index_path)
    query = st.text_input("Filter text", value="")
    contract_types = st.multiselect("Contract types", _options(df, "contract_type"))
    departments = st.multiselect("Departments", _options(df, "department"))
    counterparties = st.multiselect("Counterparties", _options(df, "counterparty_name"))
    statuses = st.multiselect("Statuses", _options(df, "status"))

    # Decoded boolean/risk filters (yes/no/na labels from raw-first extraction).
    # Data-driven: only offer filters for decoded fields actually present.
    DECODED_FILTER_COLS = [
        "Over5M", "Over100M", "WithEndDate", "FlagNeedLegal", "FlagNeedGFN",
        "IsRisksAccepted", "IsAuthoritySufficient", "IsRenew",
        "PreliminaryReviewFlag", "IfRelatedToData",
        "IncludingExternalGuarantees", "iscontractfinancial",
    ]
    decoded_selections: Dict[str, List[str]] = {}
    with st.expander("Risk / approval filters (decoded)"):
        cols = st.columns(3)
        for i, field in enumerate(DECODED_FILTER_COLS):
            if field in df.columns:
                decoded_selections[field] = cols[i % 3].multiselect(
                    field, _options(df, field)
                )

    filtered = _filter_contracts(df, query, contract_types, departments, counterparties, statuses)
    for field, selected in decoded_selections.items():
        if selected:
            filtered = filtered[filtered[field].astype(str).isin(selected)]

    st.caption(f"{len(filtered)} record(s) · {filtered.shape[1]} column(s)")

    # Primary readable grid vs. full column set (all content-bearing fields).
    base_cols = [
        "id", "ref_no", "title", "counterparty_name", "department",
        "amount_label", "requested_date", "status", "contract_type_label",
    ]
    show_all = st.checkbox(
        "Show all columns (all content-bearing raw + decoded fields)", value=False
    )
    if show_all:
        display_df = filtered.drop(
            columns=["decoded_fields", "contextual_fields"], errors="ignore"
        )
    else:
        keep = [c for c in base_cols if c in filtered.columns]
        keep += [f for f in ("Over5M", "Over100M", "FlagNeedLegal", "IsRisksAccepted") if f in filtered.columns]
        display_df = filtered[keep].copy()
        # Show contract_type_label when available instead of raw code
        if "contract_type_label" in display_df.columns and "contract_type" in display_df.columns:
            display_df["contract_type"] = display_df["contract_type_label"]
            display_df = display_df.drop(columns=["contract_type_label"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Per-record detail (merged 4-tab view, same function as search results).
    with st.expander("Record detail"):
        refs = list(dict.fromkeys(
            x for x in (filtered["ref_no"].tolist() if "ref_no" in filtered.columns else [])
            if x
        ))[:200]
        if refs:
            chosen = st.selectbox("Contract ref_no", refs)
            if chosen:
                _render_contract_detail(chosen, index_path)
        else:
            st.info("No records to inspect.")



def _render_dashboard(index_path: str):
    st.subheader("Dashboard")
    df = _load_sections(index_path)
    total_contracts = df["contract_id"].nunique() if "contract_id" in df else 0
    total_chunks = len(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Contracts", total_contracts)
    with col2:
        st.metric("Chunks", total_chunks)
    with col3:
        st.metric("Departments", df["department"].nunique() if "department" in df else 0)
    with col4:
        st.metric("Counterparties", df["counterparty_name"].nunique() if "counterparty_name" in df else 0)

    st.markdown("### By contract type")
    if "contract_type" in df:
        st.bar_chart(df["contract_type"].astype(str).value_counts().sort_index())
    st.markdown("### By department")
    if "department" in df:
        st.bar_chart(df["department"].fillna("(unknown)").astype(str).value_counts().head(20))
    st.markdown("### Requested dates")
    if "requested_date" in df:
        date_series = pd.to_datetime(df["requested_date"], errors="coerce")
        st.line_chart(date_series.value_counts().sort_index())


def main():
    st.set_page_config(page_title="OA Contract Screening", layout="wide")
    st.title("OA Contract Screening")
    st.caption("Ask natural-language questions about contracts; the agentic UI routes "
               "to the right search tool and shows its reasoning.")

    index_path = st.sidebar.text_input("Index path", value=DEFAULT_INDEX_PATH)
    page = st.sidebar.radio("View", ["Ask (Agentic)", "Browse", "Dashboard"], index=0)

    if not os.path.exists(index_path):
        st.error(f"Index path not found: {index_path}")
        st.stop()

    health = check_llm_health()
    if not health.get("ok"):
        st.error(
            "LLM is down or unreachable. "
            + str(health.get("error") or "unknown error")
        )
    else:
        st.caption("LLM ready: " + chr(96) + str(health.get("model")) + chr(96))
    # Settings expander in sidebar: column picker (moved inline to save space)
    with st.sidebar.expander("Settings"):
        sidebar_cols = st.multiselect(
            "Table columns", _TABLE_BASE_COLUMNS + _TABLE_EXTRA_COLUMNS,
            default=_TABLE_BASE_COLUMNS, key="sidebar_columns_picker",
        )
        st.session_state["sidebar_columns_"] = sidebar_cols

    # On-demand live generation probe (off the per-rerun critical path).
    if st.sidebar.button("Recheck LLM (live probe)"):
        check_llm_health.clear()
        with st.spinner("Probing LLM generation..."):
            deep_health = check_llm_health(deep=True)
        if deep_health.get("ok"):
            st.sidebar.success("LLM generation OK: " + str(deep_health.get("model")))
        else:
            st.sidebar.error("LLM generation failed: "
                             + str(deep_health.get("error") or "unknown error"))
    if page == "Ask (Agentic)":
        _render_agentic(index_path)
    elif page == "Browse":
        _render_browser(index_path)
    else:
        _render_dashboard(index_path)


if __name__ == "__main__":
    main()

