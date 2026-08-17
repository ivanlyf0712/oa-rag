"""Cross-session memory (Hindsight) as one deep module.

The Hindsight port previously exposed the memory concern as three shallow
modules (hindsight_client / persona / agent_config) plus ~120 lines of glue in
the UI, and passed the per-turn outcome around as a loosely-shaped dict probed
with .get(). This module puts the whole per-turn memory flow behind a narrow
interface:

    prepare_turn(query)  -> TurnMemory   (bank, persona profile, recall gating)
    complete_turn(...)   -> MemoryOutcome (typed, explicit recall/retain state)

Internally it composes the Hindsight REST client, the CARA persona mapping,
and the on-demand recall gate. All Hindsight calls degrade gracefully, so the
search UI keeps working when Hindsight is down (off = empty bank).

The module is Streamlit-free: session caching of the bank/profile is injected
via a session dict (st.session_state in the app, a plain dict in tests).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from apps.search import hindsight_client as _hc
from apps.search.persona import DispositionProfile
from apps.search.agent_config import default_agent_config, persona_to_profile_dict


def resolve_bank(session: Dict[str, Any]) -> str:
    """Active Hindsight bank id: session override -> env -> '' (off).

    An explicit session value wins even when empty (user turned it off in
    Settings); the env var only applies until the Settings input is touched.
    """
    bank = session.get("hindsight_bank")
    if bank is None:
        bank = os.getenv("HINDSIGHT_BANK_ID") or ""
    return str(bank).strip()


def ui_url() -> str:
    """Base URL of the Hindsight Web UI (must be reachable from the browser)."""
    return (os.getenv("HINDSIGHT_UI_URL") or "http://localhost:9999").rstrip("/")


@dataclass
class TurnMemory:
    """Everything the agent/UI needs to run one turn with memory."""
    bank: str                       # "" -> memory off
    profile: DispositionProfile     # persona driving answer generation

    @property
    def enabled(self) -> bool:
        return bool(self.bank)


@dataclass
class MemoryOutcome:
    """Typed result of a completed turn's memory activity.

    recall:  "recall" | "skip" | "none"  (none when memory is off)
    retained: True/False once retain attempted; None when the turn failed
              (retain skipped) or memory is off.
    """
    bank: str = ""
    recall: str = "none"
    retained: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Shape stored on result['hindsight'] for the UI caption."""
        if not self.bank:
            return {}
        d: Dict[str, Any] = {"bank": self.bank, "recall": self.recall}
        if self.retained is not None:
            d["retained"] = self.retained
        return d


class Memory:
    """Deep cross-session memory module; one seam for the whole feature."""

    def __init__(self, session: Optional[Dict[str, Any]] = None,
                 retain_fn=None):
        # session is a mutable mapping used for per-session caching
        # (st.session_state in the app; a plain dict in tests).
        # retain_fn is the retain adapter at this seam (default: the Hindsight
        # REST client); injectable so tests drive the module through its seam.
        self._session = session if session is not None else {}
        self._retain_fn = retain_fn or _hc.retain

    # ── prepare ──────────────────────────────────────────────────
    def prepare_turn(self, query: str) -> TurnMemory:
        """Resolve the bank + persona profile for this turn.

        Bank set -> profile loaded from the bank disposition (cached per bank).
        Bank off -> profile from the local Settings-page sliders (neutral
        default appends nothing, so answers are unchanged).
        """
        bank = resolve_bank(self._session)
        if bank:
            profile = self._load_profile(bank)
        else:
            cfg = self._session.get("agent_config") or default_agent_config()
            profile = DispositionProfile.from_dict(persona_to_profile_dict(cfg["persona"]))
        return TurnMemory(bank=bank, profile=profile)

    def _load_profile(self, bank: str) -> DispositionProfile:
        cache_key = "_hindsight_profile_" + bank
        prof = self._session.get(cache_key)
        if prof is None:
            prof = DispositionProfile.from_hindsight(bank)
            self._session[cache_key] = prof
        return prof

    # ── recall gate ──────────────────────────────────────────────
    @staticmethod
    def should_recall(query: str) -> bool:
        """On-demand recall gate: only explicit cross-session references fire."""
        return _hc.needs_recall(query)

    # ── complete ─────────────────────────────────────────────────
    def complete_turn(self, query: str, rows: List[Dict[str, Any]],
                      turn: TurnMemory, *, recalled: bool,
                      turn_succeeded: bool) -> MemoryOutcome:
        """Retain a successful turn and report the typed memory outcome.

        recalled: whether the agent's ReAct loop actually fired recall.
        turn_succeeded: failed turns are not retained.
        """
        if not turn.enabled:
            return MemoryOutcome()
        outcome = MemoryOutcome(
            bank=turn.bank,
            recall="recall" if recalled else "skip",
        )
        if turn_succeeded:
            outcome.retained = bool(self._retain(query, rows, turn.bank))
        return outcome

    # ── retain internals ─────────────────────────────────────────
    def _retain(self, query: str, rows: List[Dict[str, Any]], bank: str) -> bool:
        content = self.build_retain_content(query, rows)
        tags = ["oa-rag", "search"]
        for r in rows[:5]:
            meta = r.get("metadata") or {}
            cp = meta.get("counterparty_name")
            ref = meta.get("ref_no") or r.get("ref_no")
            if cp and cp not in tags:
                tags.append(cp)
            if ref and ref not in tags:
                tags.append(ref)
        return self._retain_fn(content, bank=bank, context="oa-rag-search",
                               tags=tags, async_=True)

    @staticmethod
    def build_retain_content(query: str, rows: List[Dict[str, Any]],
                             max_rows: int = 3) -> str:
        """Compose a compact memory document for one agent turn (pure)."""
        lines = ["OA contract search turn. User query: %s" % (query or "")]
        for r in rows[:max_rows]:
            meta = r.get("metadata") or {}
            ref = meta.get("ref_no") or r.get("ref_no") or ""
            cp = meta.get("counterparty_name") or ""
            title = meta.get("title") or ""
            lines.append("- %s | %s | %s" % (ref, cp, title))
        return "\n".join(lines)

    # ── graph preview (UI) ───────────────────────────────────────
    @staticmethod
    def graph_stats(bank: str) -> Dict[str, Any]:
        return _hc.get_entity_graph(bank, limit=50)
