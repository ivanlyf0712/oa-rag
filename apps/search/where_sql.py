"""contracts_where SQL translation (ticket 05) — pure, unit-tested.

Ported from corpchat search/tools.py (search_messages_where): a natural
language condition ("contracts over HK$5M ending before 2027", "contracts
needing legal review") is translated to a validated read-only SELECT over
the sections SQLite table. Rule-based translation first (deterministic),
LLM text-to-SQL as fallback, strict validation either way. Translation
returns None when it cannot produce safe SQL; callers then fall back to an
index scan (semantic search over the same corpus) — never an unhandled
exception to the agent.

The sections table stores every contract chunk as (id, text, tags) where
tags is a JSON string; structured fields are read via json_extract.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger("oa-where-sql")

SQL_TABLES = ("sections",)
# Safety cap on scanned sections (several chunks per contract).
MAX_WHERE_SECTIONS = 5000

_SQL_SELECT = "SELECT id, text, tags FROM sections"

_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "grant", "create",
    "truncate", "attach", "pragma", "replace", "--", "/*",
)

# Phrases that enumerate rather than filter ("list all contracts ...").
_ENUM_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?(?:list|show|display|give\s+me|get|find)\s+",
    re.IGNORECASE,
)
_ENUM_FILLER_RE = re.compile(
    r"^\s*(?:(?:all|every|the|contracts?|with|where|that|which|having|"
    r"matching|have|has|are)(?:\s+|$))",
    re.IGNORECASE,
)


def enumeration_remainder(condition: str) -> str:
    """Strip enumeration boilerplate, returning the actual filter content.

    "list all contracts" -> "" (no filter; callers return every contract).
    "list all contracts with risk not accepted" -> "risk not accepted".
    """
    c = (condition or "").strip()
    c = _ENUM_PREFIX_RE.sub("", c)
    for _ in range(4):
        stripped = _ENUM_FILLER_RE.sub("", c)
        if stripped == c:
            break
        c = stripped
    return c.strip()


def _validate_sql(sql: str) -> Optional[str]:
    """Validate a generated SELECT: single statement, read-only, sections only.

    Returns the cleaned SQL (with a LIMIT guard) or None when unsafe.
    """
    if not sql:
        return None
    s = sql.strip().rstrip(";").strip()
    low = s.lower()
    if not low.startswith("select"):
        return None
    if ";" in s:
        return None
    if any(bad in low for bad in _FORBIDDEN):
        return None
    tables = re.findall(r"\bfrom\s+(\w+)", low) + re.findall(r"\bjoin\s+(\w+)", low)
    if not tables or any(t not in SQL_TABLES for t in tables):
        return None
    if "limit" not in low:
        s += " LIMIT %d" % MAX_WHERE_SECTIONS
    return s


# ── rule-based condition -> SQL ─────────────────────────────────────

# Threshold flag phrases (checked before the amount rule so "over 5m" keeps
# the form-flag semantics from risk_search._KEYWORD_FILTERS).
_THRESHOLD_FLAG_PHRASES = ("over 5m", "over5m", "over 100m")

_AMOUNT_RE = re.compile(
    r"(over|above|more\s+than|greater\s+than|exceeding|at\s+least|"
    r"under|below|less\s+than|at\s+most|up\s+to)\s*"
    r"(hk\$|hkd|\$)?\s*(\d+(?:\.\d+)?)\s*(million|m|thousand|k)?\b",
    re.IGNORECASE,
)
_AMOUNT_OPS = {
    "over": ">", "above": ">", "more than": ">", "greater than": ">",
    "exceeding": ">",
    "at least": ">=",
    "under": "<", "below": "<", "less than": "<",
    "at most": "<=", "up to": "<=",
}
_AMOUNT_MULT = {"m": 1_000_000, "million": 1_000_000, "k": 1_000, "thousand": 1_000}

_DATE_RE = re.compile(
    r"(ending|expiring|expires?|end|starting|starts?|signed|requested|created)\s+"
    r"(on\s+or\s+before|on\s+or\s+after|before|after|by)\s+"
    r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?",
    re.IGNORECASE,
)
_DATE_FIELDS = {
    "ending": "contract_end_date", "expiring": "contract_end_date",
    "expire": "contract_end_date", "expires": "contract_end_date",
    "end": "contract_end_date",
    "starting": "contract_start_date", "start": "contract_start_date",
    "starts": "contract_start_date",
    "signed": "requested_date", "requested": "requested_date",
    "created": "requested_date",
}

# Risk-accepted phrases need explicit handling ("risk accepted" vs
# "risk not accepted"); the other coded flags come from _KEYWORD_FILTERS.
_RISK_ACCEPTED_RULES = (
    (("risk not accepted", "risks not accepted", "risk unaccepted",
      "unaccepted risk", "unaccepted risks"), "IsRisksAccepted", "no"),
    (("risk accepted", "risks accepted"), "IsRisksAccepted", "yes"),
)

# Status words for rule-based matching live in the filters module (single
# source of truth for the status vocabulary); imported here as _STATUS_WORDS.
from apps.search.filters import STATUS_WORDS as _STATUS_WORDS


def _json_label_cond(field: str, value: str) -> str:
    return ("LOWER(json_extract(tags, '$.decoded_fields.%s.label')) = '%s'"
            % (field, value))


def _amount_sql(m: "re.Match") -> Optional[str]:
    comp, currency, number, suffix = m.groups()
    if not currency and not suffix:
        try:
            if float(number) < 10_000:
                return None  # bare small number: too ambiguous for an amount
        except ValueError:
            return None
    op = _AMOUNT_OPS[" ".join(comp.lower().split())]
    mult = _AMOUNT_MULT.get((suffix or "").lower(), 1)
    value = float(number) * mult
    value = int(value) if value == int(value) else value
    return ("CAST(json_extract(tags, '$.amount') AS REAL) %s %s" % (op, value))


def _date_sql(m: "re.Match") -> str:
    noun, rel, year, month, day = m.groups()
    field = _DATE_FIELDS[noun.lower()]
    rel = " ".join(rel.lower().split())
    if month and day:
        bound = "%s-%s-%s" % (year, month, day)
        op = {"before": "<", "after": ">",
              "on or before": "<=", "on or after": ">=", "by": "<="}[rel]
    else:
        bound, op = {
            "before": ("%s-01-01" % year, "<"),
            "by": ("%s-12-31" % year, "<="),
            "on or before": ("%s-12-31" % year, "<="),
            "after": ("%s-12-31" % year, ">"),
            "on or after": ("%s-01-01" % year, ">="),
        }[rel]
    return "date(json_extract(tags, '$.%s')) %s date('%s')" % (field, op, bound)


def _condition_to_sql(condition: str) -> Optional[str]:
    """Deterministic natural-language condition -> SQL (no LLM).

    Rule order: threshold flag phrases, amount comparisons, date bounds,
    coded-flag labels, status words. Returns None when no rule matches.
    """
    c = (condition or "").lower()
    if not c:
        return None

    from apps.risk_search import _KEYWORD_FILTERS

    for phrase in _THRESHOLD_FLAG_PHRASES:
        if phrase in c and not re.search(r"(hk\$|hkd|\$)\s*" + re.escape(phrase.split()[-1]), c):
            field = "Over5M" if "5m" in phrase else "Over100M"
            return "%s WHERE %s LIMIT %d" % (
                _SQL_SELECT, _json_label_cond(field, "yes"), MAX_WHERE_SECTIONS)

    m = _AMOUNT_RE.search(c)
    if m:
        cond = _amount_sql(m)
        if cond:
            return "%s WHERE %s LIMIT %d" % (_SQL_SELECT, cond, MAX_WHERE_SECTIONS)

    m = _DATE_RE.search(c)
    if m:
        return "%s WHERE %s LIMIT %d" % (_SQL_SELECT, _date_sql(m), MAX_WHERE_SECTIONS)

    for phrases, field, value in _RISK_ACCEPTED_RULES:
        if any(p in c for p in phrases):
            return "%s WHERE %s LIMIT %d" % (
                _SQL_SELECT, _json_label_cond(field, value), MAX_WHERE_SECTIONS)

    for phrase, field, value in sorted(_KEYWORD_FILTERS, key=lambda t: -len(t[0])):
        if phrase in c:
            return "%s WHERE %s LIMIT %d" % (
                _SQL_SELECT, _json_label_cond(field, value), MAX_WHERE_SECTIONS)

    for word in _STATUS_WORDS:
        if re.search(r"\b%s\b" % word, c):
            return ("%s WHERE LOWER(json_extract(tags, '$.status_label')) "
                    "LIKE '%s%%' LIMIT %d" % (_SQL_SELECT, word, MAX_WHERE_SECTIONS))
    if re.search(r"\bexpired\b", c):
        return ("%s WHERE date(json_extract(tags, '$.contract_end_date')) "
                "< date('now') LIMIT %d" % (_SQL_SELECT, MAX_WHERE_SECTIONS))
    return None


# ── LLM text-to-SQL fallback ────────────────────────────────────────

_TEXT_TO_SQL_SYSTEM = (
    "You convert a natural-language filter into a single SQLite SELECT over "
    "the table sections(id TEXT, text TEXT, tags TEXT). tags is a JSON string; "
    "read fields with json_extract(tags, '$.key'). Useful keys: ref_no, title, "
    "counterparty_name, department, contract_type, status_label, amount (REAL), "
    "contract_start_date, contract_end_date, requested_date, and decoded_fields "
    "(JSON object of coded flags, e.g. "
    "json_extract(tags, '$.decoded_fields.FlagNeedLegal.label') = 'yes'). "
    "Reply with ONLY one SQL statement of the form "
    "SELECT id, text, tags FROM sections WHERE <condition>. "
    "Read-only SELECT, no semicolons, no markdown, no explanation."
)


def _llm_condition_to_sql(condition: str, client: Any = None) -> Optional[str]:
    """LLM fallback: generate a sections SELECT, strictly validated.

    Returns None on any failure (no client, empty reply, unsafe SQL).
    """
    try:
        if client is None:
            from apps.search.litellm_client import LiteLLMClient
            client = LiteLLMClient()
        result = client.chat(
            [
                {"role": "system", "content": _TEXT_TO_SQL_SYSTEM},
                {"role": "user", "content": condition},
            ],
            temperature=0.0, max_tokens=200, timeout=10,
        )
    except Exception as e:
        logger.warning("Text-to-SQL failed: %s", e)
        return None
    # Strip accidental markdown fences before validation.
    text = (result or "").strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return _validate_sql(text)


def condition_to_sql(condition: str, llm_client: Any = None,
                     allow_llm: bool = True) -> Optional[str]:
    """Full translation chain: enumeration strip -> rules -> LLM fallback.

    Returns (sql) or None when translation is impossible. An empty filter
    remainder (bare "list all") returns the sentinel ALL_CONTRACTS_SQL so
    callers can distinguish "no filter" from "untranslatable".
    """
    cond = (condition or "").strip()
    remainder = enumeration_remainder(cond)
    if not remainder:
        return "%s LIMIT %d" % (_SQL_SELECT, MAX_WHERE_SECTIONS)
    sql = _condition_to_sql(remainder)
    if sql is not None:
        return sql
    if not allow_llm:
        return None
    return _llm_condition_to_sql(remainder, client=llm_client)

