#!/usr/bin/env python3
"""Risky-contract search planner and scoring (RISK_SEARCH_SPEC).

Architecture:
  - LLM is used ONLY for intent detection + filter extraction (RiskPlanner).
  - Filtering, scoring, severity, gating and sorting are deterministic backend
    rules operating on the decoded yes/no/na contract fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("oa-risk-search")

# ═══════════════════════════════════════════════════════════════════
# Ticket 1 — Planner contract
# ═══════════════════════════════════════════════════════════════════

MODE_RISKY = "risky_contracts"
MODE_GENERAL = "general_search"
MODE_CLARIFY = "clarify"
VALID_MODES = (MODE_RISKY, MODE_GENERAL, MODE_CLARIFY)

# v1 allowlist of filterable risk fields (decoded yes/no/na columns).
# Expanded (Ticket 9) to cover the full set of content-bearing risk tags.
# NOTE: Solely, IsMC, IsPreAuthoritySufficient and the PromptFor* flags are
# always-NULL in the source data and are therefore not present in the index.
RISK_FILTER_FIELDS: Tuple[str, ...] = (
    # core outcome / authority
    "IsRisksAccepted",
    "IsAuthoritySufficient",
    # need-approval flags
    "FlagNeedLegal",
    "FlagNeedGFN",
    # threshold / value flags
    "Over5M",
    "Over100M",
    "WithEndDate",
    "Saved",
    "IncludingExternalGuarantees",
    "IsRenew",
    "iscontractfinancial",
    # review flags
    "PreliminaryReviewFlag",
    "preliminaryreviewflag2",
    "needapreliminaryreviewbygroupl",
    # related-party / data / capex flags
    "IfRelatedToData",
    "relatedtocapexpropertyleasingc",
    "generalpurchaseandoverhk50k",
    "unlimitedliabilitiesorliabilit",
    # documentation completeness
    "allrelevantdocumentationhasbee",
)

ALLOWED_VALUES = ("yes", "no", "na")

# Confidence policy: high → auto-route; low → clarify.
CONFIDENCE_THRESHOLD = 0.6

DEFAULT_CLARIFICATION = (
    "Do you want me to search for risky contracts "
    "(e.g. risk not accepted, legal review needed), or perform a general contract search?"
)


def normalize_filter_value(value: Any) -> Optional[str]:
    """Normalize a filter value to yes/no/na; return None if unrecognised."""
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "yes": "yes", "y": "yes", "true": "yes", "1": "yes", "required": "yes", "needed": "yes",
        "no": "no", "n": "no", "false": "no", "0": "no", "not": "no", "not accepted": "no",
        "unaccepted": "no", "rejected": "no",
        "na": "na", "n/a": "na", "2": "na", "none": "na", "unknown": "na",
    }
    return mapping.get(text)


def validate_filters(filters: Any) -> List[Dict[str, str]]:
    """Validate planner filter clauses against the allowlist (Ticket 1).

    Unknown fields are dropped (rejected); values are normalized to yes/no/na.
    Returns a clean list of {"field", "op", "value"} clauses.
    """
    clean: List[Dict[str, str]] = []
    if not isinstance(filters, list):
        return clean
    for clause in filters:
        if not isinstance(clause, dict):
            continue
        field = clause.get("field")
        if field not in RISK_FILTER_FIELDS:
            logger.debug("rejecting unknown risk filter field: %r", field)
            continue
        value = normalize_filter_value(clause.get("value"))
        if value is None:
            continue
        clean.append({"field": field, "op": "=", "value": value})
    return clean


# ═══════════════════════════════════════════════════════════════════
# Ticket 3 — Deterministic risk filter application
# ═══════════════════════════════════════════════════════════════════

def apply_risk_filters(df: pd.DataFrame, filters: List[Dict[str, str]]) -> pd.DataFrame:
    """Apply validated filter clauses to the decoded columns of `df`.

    Deterministic exact matching on yes/no/na labels — no semantic similarity.
    """
    out = df
    for clause in validate_filters(filters):
        field, value = clause["field"], clause["value"]
        if field not in out.columns:
            continue
        out = out[out[field].fillna("").astype(str).str.lower() == value]
    return out


# ═══════════════════════════════════════════════════════════════════
# Ticket 4 — Weighted risk scoring + severity tiers
# ═══════════════════════════════════════════════════════════════════

def _default_weights() -> Dict[Tuple[str, str], int]:
    """Default risk weights (Ticket 9, expanded).

    Grouped by risk significance. `WithEndDate` and `Saved` are workflow/state
    fields (mostly yes / all yes) and carry no risk weight.
    """
    return {
        # core outcome / authority (highest signal)
        ("IsRisksAccepted", "no"): 50,
        ("IsAuthoritySufficient", "no"): 20,
        # need-approval flags
        ("FlagNeedLegal", "yes"): 20,
        ("FlagNeedGFN", "yes"): 15,
        # related-party / data / capex / liability flags
        ("IfRelatedToData", "yes"): 15,
        ("unlimitedliabilitiesorliabilit", "yes"): 15,
        ("relatedtocapexpropertyleasingc", "yes"): 10,
        ("generalpurchaseandoverhk50k", "yes"): 10,
        # threshold / value flags
        ("Over5M", "yes"): 10,
        ("Over100M", "yes"): 10,
        ("IncludingExternalGuarantees", "yes"): 10,
        ("iscontractfinancial", "yes"): 5,
        ("IsRenew", "yes"): 5,
        # review flags
        ("PreliminaryReviewFlag", "yes"): 10,
        ("preliminaryreviewflag2", "yes"): 5,
        ("needapreliminaryreviewbygroupl", "yes"): 5,
        # documentation completeness
        ("allrelevantdocumentationhasbee", "no"): 15,
    }


def load_risk_weights() -> Dict[Tuple[str, str], int]:
    """Load weights from RISK_WEIGHTS_JSON (config) or fall back to defaults.

    Override format: {"Field|value": points}, e.g. {"IsRisksAccepted|no": 60}.
    """
    raw = os.getenv("RISK_WEIGHTS_JSON", "").strip()
    weights = _default_weights()
    if raw:
        try:
            overrides = json.loads(raw)
            for key, pts in overrides.items():
                field, _, value = key.partition("|")
                if field and value:
                    weights[(field, normalize_filter_value(value) or value)] = int(pts)
        except Exception as e:
            logger.warning("invalid RISK_WEIGHTS_JSON, using defaults: %s", e)
    return weights


DEFAULT_SEVERITY_TIERS: List[Tuple[str, int]] = [("high", 50), ("medium", 20), ("low", 0)]


def load_severity_tiers() -> List[Tuple[str, int]]:
    """Load severity thresholds from RISK_SEVERITY_JSON, e.g. {"high": 50, "medium": 20}."""
    raw = os.getenv("RISK_SEVERITY_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            tiers = sorted(((name, int(thr)) for name, thr in parsed.items()),
                           key=lambda t: -t[1])
            if tiers:
                return tiers
        except Exception as e:
            logger.warning("invalid RISK_SEVERITY_JSON, using defaults: %s", e)
    return list(DEFAULT_SEVERITY_TIERS)


def _severity_for(score: int, tiers: List[Tuple[str, int]]) -> str:
    for name, threshold in tiers:
        if score >= threshold:
            return name
    return tiers[-1][0] if tiers else "low"


def _row_decoded_label(row: pd.Series, field: str) -> Optional[str]:
    """Read a yes/no/na label for `field` from a row (flat column or decoded_fields)."""
    if field in row.index:
        value = row.get(field)
        if value is not None and str(value).strip() != "":
            return normalize_filter_value(value)
    decoded = row.get("decoded_fields")
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except Exception:
            decoded = None
    if isinstance(decoded, dict):
        pair = decoded.get(field) or {}
        return normalize_filter_value(pair.get("label"))
    return None


def score_risk(
    df: pd.DataFrame,
    weights: Optional[Dict[Tuple[str, str], int]] = None,
    tiers: Optional[List[Tuple[str, int]]] = None,
) -> pd.DataFrame:
    """Compute risk_score, risk_severity, matched_signals, risk_explanation per row."""
    weights = weights if weights is not None else load_risk_weights()
    tiers = tiers if tiers is not None else load_severity_tiers()

    scores, severities, signals_col, explanations = [], [], [], []
    for _, row in df.iterrows():
        total = 0
        signals: List[str] = []
        for (field, value), pts in weights.items():
            label = _row_decoded_label(row, field)
            # Risk-acceptance N/A (field never filled) is equivalent to "no";
            # the contract's risk posture was never assessed → treat as high-risk.
            if field == "IsRisksAccepted" and label == "na":
                label = "no"
            if label == value:
                total += pts
                signals.append(f"{field} = {value} (+{pts})")
        severity = _severity_for(total, tiers)
        explanation = (
            f"Risk score {total} ({severity}) based on: " + "; ".join(signals)
            if signals else f"Risk score {total} ({severity}); no risk signals matched."
        )
        scores.append(total)
        severities.append(severity)
        signals_col.append(signals)
        explanations.append(explanation)

    out = df.copy()
    out["risk_score"] = scores
    out["risk_severity"] = severities
    out["matched_signals"] = signals_col
    out["risk_explanation"] = explanations
    return out



# ═══════════════════════════════════════════════════════════════════
# Ticket 5 — Gating and sorting
# ═══════════════════════════════════════════════════════════════════

def _gate_threshold(tiers: List[Tuple[str, int]]) -> int:
    """Default gate = the 'medium' threshold (2nd-highest tier)."""
    if len(tiers) >= 2:
        return tiers[1][1]
    return tiers[0][1] if tiers else 0


def gate_and_sort(
    df: pd.DataFrame,
    min_score: Optional[int] = None,
    tiers: Optional[List[Tuple[str, int]]] = None,
) -> pd.DataFrame:
    """Hard-gate by minimum risk score, then sort by score descending."""
    tiers = tiers if tiers is not None else load_severity_tiers()
    if min_score is None:
        env = os.getenv("RISK_GATE_MIN_SCORE", "").strip()
        min_score = int(env) if env.isdigit() else _gate_threshold(tiers)
    gated = df[df["risk_score"] >= min_score]
    return gated.sort_values("risk_score", ascending=False, kind="stable")


# ═══════════════════════════════════════════════════════════════════
# Ticket 2 — LLM-based risk intent detection (planner)
# ═══════════════════════════════════════════════════════════════════

_LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
_LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.dchbi.app/")
_LITELLM_MODEL = os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# Explicit risk language → deterministic risky mode without an LLM call.
_RISK_KEYWORDS = (
    "risk", "risky", "unaccepted", "not accepted", "legal review", "legal approval",
    "authority insufficient", "insufficient authority", "over 5m", "over5m", "over 100m",
    "related to data", "preliminary review", "documentation incomplete",
    "incomplete documentation", "need legal", "gfn", "non-compliant", "noncompliant",
    "unlimited liabilit", "liability", "external guarantee", "guarantee",
    "capex", "property leasing", "related party", "related-party", "over hk50k",
    "financial contract", "renewal risk",
)

# Keyword phrase → (field, value) for deterministic routing (Ticket 9).
_KEYWORD_FILTERS: Tuple[Tuple[str, str, str], ...] = (
    ("legal", "FlagNeedLegal", "yes"),
    ("gfn", "FlagNeedGFN", "yes"),
    ("over 5m", "Over5M", "yes"),
    ("over5m", "Over5M", "yes"),
    ("over 100m", "Over100M", "yes"),
    ("related to data", "IfRelatedToData", "yes"),
    ("unlimited liabilit", "unlimitedliabilitiesorliabilit", "yes"),
    ("liability", "unlimitedliabilitiesorliabilit", "yes"),
    ("external guarantee", "IncludingExternalGuarantees", "yes"),
    ("guarantee", "IncludingExternalGuarantees", "yes"),
    ("capex", "relatedtocapexpropertyleasingc", "yes"),
    ("property leasing", "relatedtocapexpropertyleasingc", "yes"),
    ("over hk50k", "generalpurchaseandoverhk50k", "yes"),
    ("preliminary review", "PreliminaryReviewFlag", "yes"),
    ("authority insufficient", "IsAuthoritySufficient", "no"),
    ("insufficient authority", "IsAuthoritySufficient", "no"),
    ("financial contract", "iscontractfinancial", "yes"),
    ("documentation incomplete", "allrelevantdocumentationhasbee", "no"),
    ("incomplete documentation", "allrelevantdocumentationhasbee", "no"),
)

_PLANNER_SYSTEM = (
    "You are a query planner for a contract risk screening app. "
    "Classify the user query and output ONLY compact JSON with keys: "
    "mode (risky_contracts | general_search | clarify), confidence (0-1), "
    "filters (list of {field, op, value}), explanation (one short sentence), "
    "clarification_question (only when mode=clarify). "
    "Use mode=risky_contracts only for explicit risk/compliance language. "
    "Allowed filter fields: " + ", ".join(RISK_FILTER_FIELDS) + ". "
    "Allowed values: yes, no, na. "
    "The most important risk signal is IsRisksAccepted = no. "
    "Also relevant: need-approval (FlagNeedLegal/FlagNeedGFN), value thresholds "
    "(Over5M/Over100M), related-party/data/capex/liability (IfRelatedToData, "
    "relatedtocapexpropertyleasingc, generalpurchaseandoverhk50k, "
    "unlimitedliabilitiesorliabilit), review flags (PreliminaryReviewFlag, "
    "preliminaryreviewflag2, needapreliminaryreviewbygroupl), authority "
    "(IsAuthoritySufficient=no), and documentation completeness "
    "(allrelevantdocumentationhasbee=no). "
    "If unsure, use mode=clarify."
)



class RiskPlanner:
    """LLM-backed query planner. LLM only interprets intent; backend decides risk."""

    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = _LITELLM_MODEL):
        self.api_base = api_base or _LITELLM_BASE_URL
        self.api_key = api_key or _LITELLM_API_KEY
        self.model = model
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _call_llm(self, messages: List[Dict], max_tokens: int = 400) -> str:
        # 经由共享 LiteLLMClient (单一 LLM 入口); 失败 → 空串 → 关键词兜底。
        from apps.search.litellm_client import LiteLLMClient
        try:
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key,
                                   model=self.model)
            return client.chat(messages, temperature=0.0, max_tokens=max_tokens,
                               timeout=30,
                               response_format={"type": "json_object"}).strip()
        except Exception as e:
            logger.warning("risk planner LLM call failed: %s", e)
            return ""

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _keyword_fallback(self, query: str) -> Optional[Dict[str, Any]]:
        """Deterministic routing on explicit risk language (no LLM needed)."""
        q = query.lower()
        if not any(kw in q for kw in _RISK_KEYWORDS):
            return None
        filters: List[Dict[str, str]] = []
        if "risk" in q or "risky" in q or "not accepted" in q or "unaccepted" in q:
            # generic risk intent → default to the top signal
            filters.append({"field": "IsRisksAccepted", "op": "=", "value": "no"})
        seen = {f["field"] for f in filters}
        for phrase, field, value in _KEYWORD_FILTERS:
            if phrase in q and field not in seen:
                filters.append({"field": field, "op": "=", "value": value})
                seen.add(field)
        return {
            "mode": MODE_RISKY,
            "confidence": 0.85,
            "filters": validate_filters(filters),
            "explanation": "Query contains explicit risk language; routed to risky-contract search.",
        }

    def plan(self, query: str) -> Dict[str, Any]:
        """Return the planner contract dict for a user query."""
        query = (query or "").strip()
        if not query:
            return {"mode": MODE_CLARIFY, "confidence": 1.0, "filters": [],
                    "explanation": "Empty query.", "clarification_question": DEFAULT_CLARIFICATION}
        if query in self._cache:
            return self._cache[query]

        # 1) deterministic keyword routing for explicit risk language
        fallback = self._keyword_fallback(query)
        if fallback is not None:
            self._cache[query] = fallback
            return fallback

        # 2) LLM classification
        raw = self._call_llm([
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": f"Query: {query}\n\nJSON:"},
        ])
        parsed = self._extract_json(raw) or {}

        mode = parsed.get("mode")
        if mode not in VALID_MODES:
            mode = MODE_GENERAL
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        filters = validate_filters(parsed.get("filters"))
        explanation = str(parsed.get("explanation") or "").strip()
        clarification = str(parsed.get("clarification_question") or "").strip()

        # Low confidence or risky intent without concrete filters → clarify.
        if confidence < CONFIDENCE_THRESHOLD:
            mode = MODE_CLARIFY
        if mode == MODE_RISKY and not filters:
            mode = MODE_CLARIFY
        if mode == MODE_CLARIFY and not clarification:
            clarification = DEFAULT_CLARIFICATION

        plan: Dict[str, Any] = {
            "mode": mode,
            "confidence": confidence,
            "filters": filters,
            "explanation": explanation or "Classified by query planner.",
        }
        if mode == MODE_CLARIFY:
            plan["clarification_question"] = clarification
        self._cache[query] = plan
        return plan


# ═══════════════════════════════════════════════════════════════════
# End-to-end helper (Tickets 3+4+5 combined)
# ═══════════════════════════════════════════════════════════════════

def run_risk_search(df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
    """Apply planner filters → score → gate → sort. Returns scored rows."""
    filtered = apply_risk_filters(df, plan.get("filters") or [])
    scored = score_risk(filtered)
    return gate_and_sort(scored)



# ═══════════════════════════════════════════════════════════════════
# Ticket 9 — Contract summary on click (readable detail + LLM summary)
# ═══════════════════════════════════════════════════════════════════

# Risk tags shown in the summary panel, grouped for readability.
# (PromptForOver5M / PromptForJustificationsUnder5M / PromptForRelatedToData are
# always-NULL in the data, so their substantive counterparts are used.)
SUMMARY_RISK_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Outcome / authority", (
        "IsRisksAccepted", "IsAuthoritySufficient",
    )),
    ("Need-approval", (
        "FlagNeedLegal", "FlagNeedGFN",
    )),
    ("Value thresholds", (
        "Over5M", "Over100M", "IncludingExternalGuarantees", "iscontractfinancial", "IsRenew",
    )),
    ("Related-party / data / capex", (
        "IfRelatedToData", "relatedtocapexpropertyleasingc",
        "generalpurchaseandoverhk50k", "unlimitedliabilitiesorliabilit",
    )),
    ("Review", (
        "PreliminaryReviewFlag", "preliminaryreviewflag2", "needapreliminaryreviewbygroupl",
    )),
    ("Documentation completeness", (
        "allrelevantdocumentationhasbee",
    )),
)

# Human-readable labels for the risk tags.
_FIELD_LABELS: Dict[str, str] = {
    "IsRisksAccepted": "risks accepted",
    "IsAuthoritySufficient": "signing authority sufficient",
    "FlagNeedLegal": "legal review required",
    "FlagNeedGFN": "GFN review required",
    "Over5M": "amount over 5M",
    "Over100M": "amount over 100M",
    "IncludingExternalGuarantees": "includes external guarantees",
    "iscontractfinancial": "financial contract",
    "IsRenew": "renewal contract",
    "IfRelatedToData": "involves personal/business data",
    "relatedtocapexpropertyleasingc": "related to capex / property leasing",
    "generalpurchaseandoverhk50k": "general purchase over HK$50k",
    "unlimitedliabilitiesorliabilit": "unlimited liability exposure",
    "PreliminaryReviewFlag": "preliminary review flagged",
    "preliminaryreviewflag2": "preliminary review (stage 2) flagged",
    "needapreliminaryreviewbygroupl": "needs preliminary review by group legal",
    "allrelevantdocumentationhasbee": "all relevant documentation provided",
}

_SUMMARY_META_FIELDS: Tuple[str, ...] = (
    "ref_no", "title", "counterparty_name", "department",
    "amount_label", "requested_date", "status", "contract_type",
)


def build_contract_context(row: pd.Series) -> Dict[str, Any]:
    """Collect the explicit evidence for one contract row (Ticket 9).

    Returns metadata, per-group risk tags (with values), matched signals,
    score and severity. This is the ONLY context handed to the LLM summarizer.
    """
    context: Dict[str, Any] = {"metadata": {}, "risk_tags": {}}
    for field in _SUMMARY_META_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip() != "":
            context["metadata"][field] = str(value)

    for group, fields in SUMMARY_RISK_GROUPS:
        entries: Dict[str, str] = {}
        for field in fields:
            value = _row_decoded_label(row, field)
            if value is not None:
                entries[_FIELD_LABELS.get(field, field)] = value
        if entries:
            context["risk_tags"][group] = entries

    for key in ("risk_score", "risk_severity", "risk_explanation"):
        if key in row.index:
            value = row.get(key)
            # coerce numpy scalars (int64/float64) to native types for JSON
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    value = str(value)
            context[key] = value
    if "matched_signals" in row.index:
        context["matched_signals"] = [str(s) for s in (row.get("matched_signals") or [])]
    return context



def _deterministic_summary(context: Dict[str, Any]) -> str:
    """Fallback summary when the LLM is unavailable (fully deterministic)."""
    meta = context.get("metadata", {})
    name = meta.get("title") or meta.get("ref_no") or "This contract"
    parts = [f"{name}"]
    if meta.get("counterparty_name"):
        parts.append(f"with {meta['counterparty_name']}")
    if meta.get("department"):
        parts.append(f"({meta['department']})")
    if meta.get("amount_label"):
        parts.append(f"amount {meta['amount_label']}")
    summary = " ".join(parts) + "."

    severity = context.get("risk_severity")
    score = context.get("risk_score")
    rationale_parts: List[str] = []
    if severity is not None:
        rationale_parts.append(f"This contract is rated {severity} risk with a score of {score}.")

    risk_tags = context.get("risk_tags", {})
    key_findings: List[str] = []
    for group in ("Outcome / authority", "Need-approval", "Related-party / data / capex", "Value thresholds", "Review", "Documentation completeness"):
        entries = risk_tags.get(group, {})
        if not entries:
            continue
        if group == "Outcome / authority":
            if entries.get("risks accepted") == "no":
                key_findings.append("risk was not accepted")
            if entries.get("signing authority sufficient") == "no":
                key_findings.append("signing authority is not sufficient")
        elif group == "Need-approval":
            if entries.get("legal review required") == "yes":
                key_findings.append("legal review is required")
            if entries.get("GFN review required") == "yes":
                key_findings.append("GFN review is required")
        elif group == "Related-party / data / capex":
            if entries.get("unlimited liability exposure") == "yes":
                key_findings.append("the contract includes unlimited liability exposure")
            if entries.get("involves personal/business data") == "yes":
                key_findings.append("the contract involves personal or business data")
            if entries.get("related to capex / property leasing") == "yes":
                key_findings.append("the contract is related to capex or property leasing")
        elif group == "Value thresholds":
            if entries.get("amount over 5M") == "yes":
                key_findings.append("the amount exceeds 5M")
            if entries.get("amount over 100M") == "yes":
                key_findings.append("the amount exceeds 100M")
            if entries.get("includes external guarantees") == "yes":
                key_findings.append("it includes external guarantees")
        elif group == "Review":
            if entries.get("preliminary review flagged") == "yes" or entries.get("preliminary review (stage 2) flagged") == "yes":
                key_findings.append("it has been flagged for preliminary review")
        elif group == "Documentation completeness":
            if entries.get("all relevant documentation provided") == "no":
                key_findings.append("supporting documentation is incomplete")

    if key_findings:
        rationale_parts.append("The main reason for review is that " + "; ".join(key_findings) + ".")

    if severity in {"high", "medium"}:
        rationale_parts.append(
            "It should be tracked closely and re-examined before any renewal, follow-up action, or further commitment."
        )
    else:
        rationale_parts.append(
            "It still merits periodic checking to make sure no hidden obligations or follow-up actions are missed."
        )

    rationale_parts.append(
        "This note is based only on the recorded risk context and is meant to support legal review judgment."
    )
    return " ".join(rationale_parts)


def summarize_contract(
    row: pd.Series,
    planner: Optional["RiskPlanner"] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Summarize one contract. LLM summarizes ONLY the explicit context (Ticket 9).

    Returns {"context": ..., "summary": ..., "llm_used": bool}.
    """
    context = build_contract_context(row)
    summary = None

    if use_llm:
        planner = planner or RiskPlanner()
        system_msg = (
            "You are a legal reviewer reviewing signed contracts and assessing whether they "
            "need tracking, renewal review, or re-examination. Use ONLY the facts in the "
            "provided context. Do not repeat the individual flags verbatim and do not invent "
            "anything not explicitly listed. Write exactly one human-sounding paragraph of 3-5 "
            "sentences that explains the context and why the contract deserves attention. Focus "
            "on the practical review judgment: what the risk means, why it matters, and whether "
            "it should be tracked or checked again before further action."
        )
        user_msg = (
            "Contract context (JSON):\n" + json.dumps(context, ensure_ascii=False, indent=2) +
            "\n\nSummary:"
        )
        raw = planner._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ], max_tokens=500)
        if raw:
            summary = raw

    if summary is None:
        summary = _deterministic_summary(context)

    return {"context": context, "summary": summary, "llm_used": bool(use_llm and summary)}

