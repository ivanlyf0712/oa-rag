"""OA-RAG — LiteLLM Client
==========================
Single client for every LLM call in the project. Encapsulates the HTTP
request, error handling, token-usage tracking and availability check so
callers never duplicate the same requests.post() pattern.

Provider: the LiteLLM proxy (OpenAI-compatible /chat/completions).
There is exactly one LLM provider; when it is unreachable, callers degrade
to deterministic, non-LLM paths (router policy / heuristic summaries)
instead of failing over to a second model.
"""

from typing import Any, Dict, List, Optional

import requests

from ._core import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MAX_TOKENS,
    LITELLM_MODEL,
    LITELLM_TIMEOUT,
    logger,
)

# 进程级 token 用量累计 (成本/用量观测)。每次成功响应累加 usage;
# reset_usage() 可清零。
_USAGE_TOTAL: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def reset_usage() -> None:
    """清零进程级 token 用量累计。"""
    _USAGE_TOTAL["prompt_tokens"] = 0
    _USAGE_TOTAL["completion_tokens"] = 0
    _USAGE_TOTAL["calls"] = 0


def usage_total() -> Dict[str, int]:
    """返回进程级累计用量 (副本)。"""
    return dict(_USAGE_TOTAL)


class LiteLLMClient:
    """Thin wrapper around the LiteLLM OpenAI-compatible chat API.

    Returns None (instead of raising) on any transport/HTTP failure so
    agents can degrade gracefully to their deterministic fallback paths.
    """

    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[int] = None):
        self.api_base = (api_base or LITELLM_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else LITELLM_API_KEY
        self.model = model or LITELLM_MODEL
        self.timeout = int(timeout or LITELLM_TIMEOUT)
        # 单实例用量 (与进程级累计并存)
        self.usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    # ── usage ────────────────────────────────────────────────────
    def _record_usage(self, usage: Optional[Dict]) -> None:
        if not usage:
            return
        try:
            p = int(usage.get("prompt_tokens", 0) or 0)
            c = int(usage.get("completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        self.usage["prompt_tokens"] += p
        self.usage["completion_tokens"] += c
        self.usage["calls"] += 1
        _USAGE_TOTAL["prompt_tokens"] += p
        _USAGE_TOTAL["completion_tokens"] += c
        _USAGE_TOTAL["calls"] += 1

    # ── chat ─────────────────────────────────────────────────────
    def chat(self, messages: List[Dict], temperature: float = 0.1,
             max_tokens: Optional[int] = None,
             timeout: Optional[int] = None,
             response_format: Optional[Dict] = None) -> str:
        """Send a chat completion and return the assistant's content.

        Returns an empty string on any failure (graceful degradation).
        """
        result = self.chat_message(messages, tools=None, temperature=temperature,
                                   max_tokens=max_tokens, timeout=timeout,
                                   response_format=response_format)
        return (result or {}).get("content") or ""

    def chat_message(self, messages: List[Dict],
                     tools: Optional[List[Dict]] = None,
                     temperature: float = 0.1,
                     max_tokens: Optional[int] = None,
                     timeout: Optional[int] = None,
                     response_format: Optional[Dict] = None) -> Optional[Dict]:
        """Send a chat completion and return the full assistant message dict.

        Unlike chat(), this returns the raw message so callers can inspect
        tool_calls (native tool-calling agents). Returns None on failure.

        Returns: {"content": str | None, "tool_calls": list | None}
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or LITELLM_MAX_TOKENS,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout or self.timeout,
            )
        except requests.exceptions.RequestException as e:
            logger.warning("LLM request failed: %s", e)
            return None
        if resp.status_code != 200:
            logger.warning("LLM request failed (%s): %s", resp.status_code, resp.text[:200])
            return None
        try:
            data = resp.json()
            msg = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as e:
            logger.warning("LLM response malformed: %s", e)
            return None
        self._record_usage(data.get("usage"))
        return {
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
        }

    # ── availability ─────────────────────────────────────────────
    def is_available(self, timeout: int = 5) -> bool:
        """Fast reachability check via GET /v1/models (no generation)."""
        if not self.api_base:
            return False
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.get(f"{self.api_base}/v1/models",
                                headers=headers, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False
