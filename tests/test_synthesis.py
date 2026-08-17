"""Tests for the shared synthesis deep module (Candidate 3).

Covers the AnswerSynthesizer directly: empty-observation short-circuit, the
callable seam and the LangChain chat-model seam, evidence sent untruncated,
and the deterministic no-LLM / LLM-down fallback (never echoing the prompt or
raw evidence).

Run:
    venv/bin/python -m pytest tests/test_synthesis.py -v
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search.synthesis import (
    EMPTY_OBSERVATION_MESSAGE,
    NO_LLM_MESSAGE,
    AnswerSynthesizer,
    build_synthesis_prompt,
)


# -- prompt construction ----------------------------------------------------
def test_prompt_contains_query_tool_and_untruncated_evidence():
    obs = chr(10).join(
        "%d. [ref=R%d] snippet" % (i, i) for i in range(1, 20)  # 19 lines
    )
    prompt = build_synthesis_prompt("q", "contract_search", obs)
    assert "q" in prompt and "contract_search" in prompt
    assert "[ref=R19]" in prompt  # no truncation of the evidence
    assert "overall contract search results" in prompt


# -- empty observation short-circuit ---------------------------------------
def test_empty_observation_returns_fixed_message():
    assert AnswerSynthesizer().synthesize("q", "contract_search", "   ") == \
        EMPTY_OBSERVATION_MESSAGE
    assert EMPTY_OBSERVATION_MESSAGE == "No matching contracts were found."


# -- callable seam ----------------------------------------------------------
def test_callable_llm_seam():
    captured = {}

    def fake_llm(messages):
        captured["messages"] = messages
        return "callable summary"

    synth = AnswerSynthesizer(llm=fake_llm)
    out = synth.synthesize("find contracts", "contract_search", "1. [ref=R1] x")
    assert out == "callable summary"
    # single human message, evidence present
    assert captured["messages"][0][0] == "human"
    assert "[ref=R1]" in captured["messages"][0][1]


def test_callable_returning_empty_falls_back():
    synth = AnswerSynthesizer(llm=lambda messages: "")
    out = synth.synthesize("q", "contract_search", "1. [ref=R1] x")
    assert "unavailable" in out.lower()
    assert "1" in out  # reports the count of evidence entries


# -- LangChain chat-model seam ----------------------------------------------
class _ChatModel:
    def __init__(self, content):
        self._content = content
        self.received = None

    def invoke(self, messages, **kwargs):
        self.received = messages
        return type("R", (), {"content": self._content})()


def test_langchain_chat_model_seam():
    model = _ChatModel("chat-model summary")
    synth = AnswerSynthesizer(llm=model)
    out = synth.synthesize("q", "contract_search", "1. [ref=R1] x")
    assert out == "chat-model summary"
    assert model.received[0][0] == "human"


# -- deterministic fallback --------------------------------------------------
def test_no_llm_returns_count_summary():
    obs = chr(10).join("%d. [ref=R%d] x" % (i, i) for i in range(1, 4))
    out = AnswerSynthesizer(llm=None).synthesize("q", "contract_search", obs)
    assert "3 matching contracts" in out
    assert "unavailable" in out.lower()


def test_no_llm_no_evidence_entries_returns_generic_message():
    out = AnswerSynthesizer(llm=None).synthesize("q", "contract_search", "no brackets here")
    assert out == NO_LLM_MESSAGE


def test_llm_exception_falls_back_without_echoing_prompt():
    class _Boom:
        def invoke(self, messages, **kw):
            raise RuntimeError("provider down")

    obs = "1. [ref=R1] x"
    out = AnswerSynthesizer(llm=_Boom()).synthesize("q", "contract_search", obs)
    assert "Summarize the overall" not in out  # never echo the prompt
    assert "unavailable" in out.lower()
    assert "1" in out
