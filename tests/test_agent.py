#!/usr/bin/env python3
"""
Tests for the Agentic Intelligence Layer (agent.py).

Verifies:
  - Intent classification (rule-based + fallback to search)
  - Routing (greeting, system_info, clarify, search, fallback)
  - Graceful degradation when LLM is unavailable
  - Multi-turn context memory
  - Search integration through the agent

Run:
    /Users/ivanlee/miniconda3/envs/ocr/bin/python -m pytest tests/test_agent.py -v
"""
import json
import os
import sys
import types
import time
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import txtai

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.gen_fake_msg import CONVERSATION_TEMPLATES, CONTACTS
from apps.corpchat.agent import (
    IntentClassifier,
    Agent,
    INTENT_GREETING,
    INTENT_SYSTEM_INFO,
    INTENT_SEARCH,
    INTENT_CLARIFY,
    INTENT_FALLBACK,
)


# ── Build test index ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"


def _build_test_index():
    docs = []
    for conv in CONVERSATION_TEMPLATES:
        label = conv["label"]
        init_name = CONTACTS[conv["initiator"]]["name"]
        resp_name = CONTACTS[conv["responder"]]["name"]
        for i, (speaker_idx, text) in enumerate(conv["turns"]):
            speaker_name = CONTACTS[speaker_idx]["name"]
            doc_id = f"{label}_{i}"
            title = f"{speaker_name} ({label})"
            match_text = f"{title}\n---\n{text}"
            tags = {
                "label": label,
                "customer_name": speaker_name,
                "company": CONTACTS[speaker_idx]["company"],
                "send_time": "2026-01-01T00:00:00",
                "external_userid": speaker_name,
                "servicer_userid": resp_name if speaker_name == init_name else init_name,
                "msgid": doc_id,
                "origin": "3" if speaker_name == init_name else "5",
                "chunk_index": i,
            }
            docs.append((doc_id, match_text, json.dumps(tags, default=str)))

    embeddings = txtai.Embeddings({
        "path": EMBEDDING_MODEL, "content": True, "objects": True,
        "hybrid": True, "scoring": {"method": "bm25"},
    })
    embeddings.index(docs)
    return embeddings


@pytest.fixture(scope="session")
def embeddings():
    return _build_test_index()


@pytest.fixture
def searcher(embeddings):
    from apps.corpchat.search import Searcher
    return Searcher(embeddings)


@pytest.fixture
def classifier_llm_down():
    """IntentClassifier with LLM explicitly disabled (simulating LLM down)."""
    return IntentClassifier(lite_llm_available=False)


@pytest.fixture
def agent_llm_down(searcher, classifier_llm_down):
    """Agent with LLM disabled — tests rule-based fallback path."""
    return Agent(searcher=searcher, classifier=classifier_llm_down)


# ══════════════════════════════════ Intent Classification ═══════════════════════════════════
class TestIntentClassification:
    """Tests for IntentClassifier — rule-based first, LLM fallback, default to search."""

    def test_greeting_en(self, classifier_llm_down):
        """English greetings are classified correctly via rules."""
        assert classifier_llm_down.classify("Hi") == INTENT_GREETING
        assert classifier_llm_down.classify("hello") == INTENT_GREETING
        assert classifier_llm_down.classify("hey") == INTENT_GREETING

    def test_greeting_zh(self, classifier_llm_down):
        """Chinese greetings are classified correctly via rules."""
        assert classifier_llm_down.classify("你好") == INTENT_GREETING
        assert classifier_llm_down.classify("嗨") == INTENT_GREETING
        assert classifier_llm_down.classify("哈囉") == INTENT_GREETING

    def test_system_info_en(self, classifier_llm_down):
        """English system info queries are classified correctly."""
        assert classifier_llm_down.classify("who are you") == INTENT_SYSTEM_INFO
        assert classifier_llm_down.classify("what can you do") == INTENT_SYSTEM_INFO

    def test_system_info_zh(self, classifier_llm_down):
        """Chinese system info queries are classified correctly."""
        assert classifier_llm_down.classify("你是誰") == INTENT_SYSTEM_INFO
        assert classifier_llm_down.classify("你能做什麼") == INTENT_SYSTEM_INFO

    def test_search_explicit(self, classifier_llm_down):
        """Explicit search keywords are classified as search."""
        assert classifier_llm_down.classify("找物流報價 方案") == INTENT_SEARCH
        assert classifier_llm_down.classify("搜尋投資美國債券") == INTENT_SEARCH
        assert classifier_llm_down.classify("search for scam messages") == INTENT_SEARCH

    def test_fallback_to_search(self, classifier_llm_down):
        """When no rule matches and LLM is unavailable, defaults to search."""
        # A query that doesn't match any keyword
        result = classifier_llm_down.classify("xyzzy foobar quux")
        assert result == INTENT_SEARCH

    def test_clarify(self, classifier_llm_down):
        """Clarification requests are classified correctly."""
        assert classifier_llm_down.classify("能再說詳細一些嗎") == INTENT_CLARIFY
        assert classifier_llm_down.classify("what do you mean") == INTENT_CLARIFY

    def test_rule_speed(self, classifier_llm_down):
        """Rule-based classification must be <1ms."""
        t0 = time.perf_counter()
        for _ in range(100):
            classifier_llm_down.classify("找物流報價 方案")
        elapsed = (time.perf_counter() - t0) * 1000
        avg = elapsed / 100
        assert avg < 1.0, f"Rule classification too slow: {avg:.2f}ms avg"


# ══════════════════════════════════ Agent Routing ═══════════════════════════════════
class TestAgentRouting:
    """Tests for Agent.process() routing logic."""

    def test_greeting_routes_static(self, agent_llm_down):
        """Greeting intent returns static greeting, no search results."""
        intent, response, results = agent_llm_down.process("你好")
        assert intent == INTENT_GREETING
        assert results == []
        assert "CorpChat Intelligence" in response
        assert "search" in response.lower()

    def test_system_info_routes_static(self, agent_llm_down):
        """System info intent returns static self-description, no search results."""
        intent, response, results = agent_llm_down.process("你是誰")
        assert intent == INTENT_SYSTEM_INFO
        assert results == []
        assert "CorpChat Intelligence" in response
        assert "search" in response.lower() or "hybrid" in response.lower()

    def test_search_routes_to_searcher(self, agent_llm_down):
        """Search intent calls Searcher.search() and returns results."""
        intent, response, results = agent_llm_down.process(
            "物流報價 方案", top_k=5, use_rerank=False, expand=False, graph_expand=0
        )
        assert intent == INTENT_SEARCH
        assert len(results) > 0, "Should return search results"
        # Response should contain relevant text from results
        assert "物流" in response or "報價" in response

    def test_clarify_routes_to_clarify(self, agent_llm_down):
        """Clarification intent asks user to rephrase."""
        intent, response, results = agent_llm_down.process("能再說詳細一些嗎")
        assert intent == INTENT_CLARIFY
        assert results == []
        assert "rephrase" in response.lower() or "clarify" in response.lower() or "再" in response

    def test_fallback_routes_to_search(self, agent_llm_down):
        """Fallback intent (no keyword match) routes to search."""
        intent, response, results = agent_llm_down.process(
            "xyzzy query that doesn't match keywords", top_k=5, use_rerank=False, expand=False, graph_expand=0
        )
        assert intent == INTENT_SEARCH
        # May or may not find results, but shouldn't crash
        assert isinstance(response, str)

    def test_no_searcher_returns_error(self, classifier_llm_down):
        """When searcher is None, agent returns an error message."""
        agent = Agent(searcher=None, classifier=classifier_llm_down)
        intent, response, results = agent.process("找物流報價 方案")
        assert "not initialized" in response.lower()
        assert results == []


# ══════════════════════════════════ Graceful Degradation ═══════════════════════════════════
class TestGracefulDegradation:
    """Tests for LLM unavailability — agent must still function."""

    def test_llm_down_search_still_works(self, agent_llm_down):
        """Search works without LLM — returns formatted results."""
        intent, response, results = agent_llm_down.process(
            "投資美國債券跟藍籌股", top_k=5, use_rerank=False, expand=False, graph_expand=0
        )
        assert intent == INTENT_SEARCH
        assert len(results) > 0
        # When LLM is down, should show formatted results as fallback answer
        assert "Found" in response or "results" in response.lower()

    def test_llm_down_no_crash_on_greeting(self, agent_llm_down):
        """Greeting works without LLM."""
        intent, response, results = agent_llm_down.process("Hi")
        assert intent == INTENT_GREETING
        assert len(results) == 0

    def test_llm_down_no_crash_on_system_info(self, agent_llm_down):
        """System info works without LLM."""
        intent, response, results = agent_llm_down.process("what can you do")
        assert intent == INTENT_SYSTEM_INFO
        assert len(results) == 0

    def test_llm_down_no_results_message(self, agent_llm_down, classifier_llm_down):
        """When search returns empty results, user gets helpful message."""
        # Use a mock searcher that returns empty results to test the no-results path
        from unittest.mock import patch
        with patch.object(agent_llm_down, 'searcher') as mock_searcher:
            mock_searcher.search.return_value = []
            intent, response, results = agent_llm_down.process(
                "xyzzy_nothing_here_zzz", top_k=5, use_rerank=False, expand=False, graph_expand=0
            )
        assert intent == INTENT_SEARCH
        assert results == []
        assert "couldn't find" in response.lower() or "No relevant" in response


# ══════════════════════════════════ Multi-turn Context ═══════════════════════════════════
class TestMultiTurnContext:
    """Tests for multi-turn conversation memory."""

    def test_history_grows(self, agent_llm_down):
        """Chat history grows with each turn."""
        agent_llm_down.process("你好")
        assert len(agent_llm_down.chat_history) == 1
        agent_llm_down.process("你能做什麼")
        assert len(agent_llm_down.chat_history) == 2

    def test_history_max(self, agent_llm_down):
        """History is capped at max_history."""
        agent = Agent(searcher=agent_llm_down.searcher, classifier=agent_llm_down.classifier, max_history=2)
        for i in range(5):
            agent.process("你好")
        assert len(agent.chat_history) == 2

    def test_reset_clears_history(self, agent_llm_down):
        """reset() clears the conversation history."""
        agent_llm_down.process("你好")
        agent_llm_down.process("你是誰")
        assert len(agent_llm_down.chat_history) == 2
        agent_llm_down.reset()
        assert len(agent_llm_down.chat_history) == 0


# ══════════════════════════════════ Search Quality ═══════════════════════════════════
class TestSearchQuality:
    """Tests that agent-mediated search returns relevant results."""

    def test_logistics_query(self, agent_llm_down):
        """Agent-mediated search for logistics returns correct message."""
        intent, response, results = agent_llm_down.process(
            "物流報價 方案", top_k=5, use_rerank=False, expand=False, graph_expand=0
        )
        assert intent == INTENT_SEARCH
        top_texts = [r.get("text", "") for r in results[:3]]
        assert any("物流" in t and "報價" in t for t in top_texts), (
            f"Logistics message not in top results: {top_texts}"
        )

    def test_investment_query(self, agent_llm_down):
        """Agent-mediated search for investment returns correct message."""
        intent, response, results = agent_llm_down.process(
            "投資美國債券跟藍籌股", top_k=5, use_rerank=False, expand=False, graph_expand=0
        )
        assert intent == INTENT_SEARCH
        top_texts = [r.get("text", "") for r in results[:3]]
        assert any("債券" in t and "藍籌" in t for t in top_texts), (
            f"Investment message not in top results: {top_texts}"
        )

    def test_label_filter_through_agent(self, agent_llm_down):
        """Agent passes label_filter through to Searcher.search()."""
        intent, response, results = agent_llm_down.process(
            "物流", top_k=10, use_rerank=False, expand=False, graph_expand=0,
            label_filter="product_inquiry"
        )
        assert intent == INTENT_SEARCH
        for r in results:
            metadata = r.get("metadata", {})
            assert metadata.get("label") == "product_inquiry", (
                f"Label filter leaked: {metadata.get('label')}"
            )


# ══════════════════════════════════ Performance ═══════════════════════════════
class TestAgentPerformance:
    """Tests for agent overhead performance."""

    def test_intent_classification_under_1ms(self, classifier_llm_down):
        """Rule-based intent classification must be <1ms."""
        queries = ["Hi", "你好", "找物流", "what can you do", "你是誰", "xyzzy query"]
        t0 = time.perf_counter()
        for _ in range(100):
            for q in queries:
                classifier_llm_down.classify(q)
        elapsed = (time.perf_counter() - t0) * 1000
        avg = elapsed / (100 * len(queries))
        assert avg < 1.0, f"Average classification too slow: {avg:.2f}ms"

    def test_agent_overhead_excludes_search(self, agent_llm_down):
        """Agent overhead (excluding search) must be <500ms for rule-based classify."""
        # Time just the classify + routing (no search)
        classifier = agent_llm_down.classifier
        t0 = time.perf_counter()
        for _ in range(100):
            classifier.classify("Hi")
        elapsed = (time.perf_counter() - t0) * 1000
        avg = elapsed / 100
        assert avg < 500.0, f"Agent overhead too slow: {avg:.2f}ms"

    def test_greeting_response_time(self, agent_llm_down):
        """Greeting routing should be instant (<10ms)."""
        t0 = time.perf_counter()
        agent_llm_down.process("Hi")
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 10.0, f"Greeting too slow: {elapsed:.1f}ms"
