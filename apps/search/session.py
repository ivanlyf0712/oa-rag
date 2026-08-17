"""Search session: agent construction + cross-session memory, Streamlit-free.

The Streamlit shell (app.py) previously mixed agent construction, the memory
bridge, and rendering in one module, so the wiring was only testable through a
mocked-Streamlit harness. This module owns the non-rendering concern behind a
small interface so a plain pytest can drive it.

Deep module: one function (build_agent) hides the provider-import fallback,
the LLM construction, the persona/memory resolution, and the LangChain-vs-
manual-router agent selection. UI notifications are injected via  so
nothing here imports Streamlit.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from apps.search.memory import Memory


def build_agent(contract_tool, where_tool, memory: Memory,
                *, notify: Optional[Callable[[str], None]] = None,
                api_key: Optional[str] = None):
    """Build the tool-calling agent over the existing pipelines.

    Falls back to the manual-ReAct CrossTableAgent only when the LLM provider
    is genuinely unavailable or misconfigured; otherwise prefers LangChain.

    notify: optional sink for user-facing degradation messages (st.caption in
    the app; a list.append or None in tests).
    """
    def _notify(msg: str) -> None:
        if notify:
            notify(msg)

    # Cross-session memory: resolve the active bank + persona profile once so
    # both agent implementations receive the same profile and hindsight_bank.
    turn = memory.prepare_turn("")
    bank = turn.bank
    profile = turn.profile

    from apps.search.agent import CrossTableAgent

    try:
        from apps.search.langchain_agent import (
            LangChainAgent, build_default_llm, AgentConfigError,
        )
    except Exception as e:  # import failure -> manual agent
        _notify(f"LangChain agent unavailable (import failed: {e}); using built-in router agent.")
        return CrossTableAgent(contract_tool=contract_tool, where_tool=where_tool,
                               profile=profile, hindsight_bank=bank or None)

    try:
        llm = build_default_llm(api_key=api_key if api_key is not None
                                else os.getenv("LITELLM_API_KEY", ""))
    except AgentConfigError as e:
        _notify(f"LangChain agent unavailable ({e}); using built-in router agent.")
        return CrossTableAgent(contract_tool=contract_tool, where_tool=where_tool,
                               profile=profile, hindsight_bank=bank or None)

    return LangChainAgent(contract_tool=contract_tool, where_tool=where_tool, llm=llm,
                          profile=profile, hindsight_bank=bank or None)
