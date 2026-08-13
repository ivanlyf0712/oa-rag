"""Shared test fixtures.

Hermetic suite: tests never call the real LiteLLM proxy. Patching
LiteLLMClient.chat_message to return None makes every LLM call site
degrade to its deterministic fallback (keyword planner, router policy,
raw observations) — exactly the paths these tests assert on.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_live_llm_calls(monkeypatch):
    from apps.search.litellm_client import LiteLLMClient
    monkeypatch.setattr(LiteLLMClient, "chat_message", lambda self, *a, **k: None)
