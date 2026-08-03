#!/usr/bin/env python3
"""
Simple AI Agent for chat-based conversation generation.
Supports Ollama (local) and LiteLLM (remote API proxy).
Set env var LLM_ENDPOINT to switch endpoints:
  - "ollama" (default) → http://127.0.0.1:11434/api/chat
  - A URL like "http://your-litellm:8000" → http://your-litellm:8000/chat/completions
Set env var LLM_MODEL to override the model name.
"""
import os
import requests
from typing import Optional

# ── Config from environment ──
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "ollama").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b").strip()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()


class AIAgent:
    """
    A lightweight chat agent that talks to an LLM backend.
    Used for generating realistic conversation threads.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        ephemeral_system_prompt: Optional[str] = None,
        quiet_mode: bool = False,
        temperature: float = 0.7,
    ):
        self.model = model or LLM_MODEL
        self.temperature = temperature
        self.quiet_mode = quiet_mode
        self.system_prompt = ephemeral_system_prompt or "You are a helpful assistant."
        self.history: list[dict[str, str]] = []

        # Determine endpoint type
        self.endpoint_type = "ollama" if LLM_ENDPOINT.lower() in ("ollama", "") else "openai"

        if self.endpoint_type == "ollama":
            self.base_url = "http://127.0.0.1:11434"
            self.chat_url = f"{self.base_url}/api/chat"
        else:
            # LiteLLM / OpenAI-compatible endpoint
            self.base_url = LLM_ENDPOINT.rstrip("/")
            self.chat_url = f"{self.base_url}/chat/completions"

    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def chat(self, user_text: str) -> str:
        """Send a message and get the model's response."""
        messages = self._build_messages(user_text)

        if self.endpoint_type == "ollama":
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 1024,
                },
            }
        else:
            # OpenAI / LiteLLM compatible format
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 1024,
                "stream": False,
            }

        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        try:
            resp = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            if self.endpoint_type == "ollama":
                reply = data.get("message", {}).get("content", "").strip()
            else:
                # OpenAI format
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Keep history for context
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            # Trim history to avoid context overflow (keep last 8 exchanges)
            if len(self.history) > 16:
                self.history = self.history[-16:]
            return reply

        except requests.exceptions.Timeout:
            return "（回應逾時，請稍後再試。）"
        except Exception as e:
            return f"（錯誤：{e}）"

    def reset(self):
        """Clear conversation history."""
        self.history = []
