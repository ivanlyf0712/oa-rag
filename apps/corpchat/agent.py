#!/usr/bin/env python3
"""
CorpChat Agentic Intelligence Layer
=====================================

Purpose:
    Add an intelligent agent layer on top of the existing Searcher, enabling
    intent classification, routing, and self-description — without modifying
    search.py.

Design (per grill session):
    - 5 intent categories: greeting, system_info, search, clarify, fallback
    - Classification: rule-based first (<1ms) → LLM fallback (2s timeout)
      → default to "search" when LLM unavailable (safe degradation)
    - Routing:
        greeting     → static greeting message
        system_info  → static self-description
        search       → call Searcher.search() with enhancement params from caller
        clarify      → ask user to rephrase
        fallback     → treat as search (safe default)
    - Multi-turn context: simple last-N turns history
    - Performance: rules <1ms, LLM <2s, total agent overhead <500ms (excluding search)

Integration:
    The agent is used by app.py's chat flow and can also be called from CLI:
        from apps.corpchat.agent import Agent, load_agent
        agent = load_agent()
        intent, response, search_results = agent.process("物流報價 方案", top_k=5)
"""

import os
import re
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("corpchat-agent")

# ── Configuration ────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import sys as _sys
if ROOT_DIR not in _sys.path:
    _sys.path.insert(0, ROOT_DIR)

# Import search layer (no modifications to search.py)
from apps.corpchat.search import (
    Searcher,
    QueryExpander,
    Reranker,
    AgenticDecider,
    load_index,
    DEFAULT_INDEX_PATH,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
)

# ── Intent categories ────────────────────────────────────────────────────
INTENT_GREETING = "greeting"
INTENT_SYSTEM_INFO = "system_info"
INTENT_SEARCH = "search"
INTENT_CLARIFY = "clarify"
INTENT_FALLBACK = "fallback"

# ── Rule-based keyword sets ──────────────────────────────────────────────
# Each set is checked first (O(n) substring match, <1ms).
# If no rule matches → LLM fallback. If LLM unavailable → default to "search".
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "yo", "嗨", "你好", "好嗎", "哈囉", "早安", "午安",
    "晚安", "久久", "怎麼樣", "最近怎麼樣", "nice to meet you",
]

SYSTEM_INFO_KEYWORDS = [
    "你是誰", "你是誰", "what is your name", "who are you", "叫什麼名字",
    "能做什麼", "what can you do", "can you help", "功能", "能力",
    "作用", "什麼功能", "多少功能", "help", "幫助", "使用說明",
    "搜索範圍", "scope", "what can you search", "can you search",
    "你能搜尋", "搜什麼", "資料範圍",
]

CLARIFY_KEYWORDS = [
    "能再說", "再說一遍", "不是很懂", "不太清楚", "clarify", "explain more",
    "詳細一些", "細節", "what do you mean", "什麼意思", "不太明白",
    "再解釋", "不太理解", "看不懂", "具體點",
]

# LLM classification timeout (seconds) — per design spec §2.7
LLM_INTENT_TIMEOUT = 2.0


class IntentClassifier:
    """
    Intent classification using rule-based first, LLM fallback.

    Classification flow:
        1. Rule matching (keywords) — <1ms, catches 80% of common cases
        2. LLM classification (if available) — <2s, handles 20% edge cases
        3. Default to "search" — when LLM is unavailable (safe degradation)

    Intent categories:
        greeting, system_info, search, clarify, fallback

    The fallback → search mapping is a safe design choice: if we can't
    classify the intent, it's better to try a search than to refuse to act.
    """

    # Static LLM classification prompt
    _LLM_PROMPT = (
        "Classify the user's intent into ONE of: greeting, system_info, search, clarify, fallback. "
        "greeting = casual hello, system_info = asking about the system's capabilities/identity, "
        "search = looking for information in chat messages, clarify = asking for more detail/explanation. "
        "If unsure, return 'fallback'. "
        "Reply with ONLY the category name."
    )

    def __init__(self, lite_llm_available: Optional[bool] = None):
        """
        Args:
            lite_llm_available: If None, auto-detect. If False, rules-only.
        """
        self._llm_check_done = False
        self._llm_available = lite_llm_available

    def _check_llm(self) -> bool:
        """Check if LiteLLM endpoint is reachable (cached)."""
        if self._llm_check_done:
            return self._llm_available if self._llm_available is not None else False

        if self._llm_available is not None:
            self._llm_check_done = True
            return self._llm_available

        if not LITELLM_API_KEY:
            self._llm_check_done = True
            self._llm_available = False
            return False

        try:
            import requests
            resp = requests.get(
                LITELLM_BASE_URL.rstrip("/") + "/v1/models",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                timeout=3,
            )
            self._llm_available = resp.status_code == 200
        except Exception:
            self._llm_available = False

        self._llm_check_done = True
        return self._llm_available

    def _rule_classify(self, query: str) -> Optional[str]:
        """
        Rule-based classification using keyword matching.
        Returns intent string or None if no rule matches.
        Complexity: O(n * m) where n=len(keywords), m=len(query) — <1ms.
        """
        q_lower = query.lower().strip()

        # System info: asking "what can you do" type questions
        # Check BEFORE greeting because "what can you do" contains "yo" (in "you")
        for kw in SYSTEM_INFO_KEYWORDS:
            if kw in q_lower:
                return INTENT_SYSTEM_INFO

        # Greeting: typically short, single word
        if len(q_lower) <= 15:
            for kw in GREETING_KEYWORDS:
                if kw in q_lower:
                    return INTENT_GREETING

        # Clarify: asking for more detail
        for kw in CLARIFY_KEYWORDS:
            if kw in q_lower:
                return INTENT_CLARIFY

        # Explicit search intent keywords
        search_kws = ["找", "搜尋", "搜索", "查", "查詢", "找找", "搜", "找一下",
                       "search", "find", "查一下", "幫我找", "協助搜尋"]
        q_words = q_lower.split()
        if any(kw in q_lower for kw in search_kws):
            return INTENT_SEARCH

        return None

    def _llm_classify(self, query: str) -> Optional[str]:
        """
        LLM-based classification as fallback.
        2s timeout per design spec. Returns None if LLM unavailable or fails.
        """
        if not self._check_llm():
            return None

        try:
            import requests
            resp = requests.post(
                LITELLM_BASE_URL.rstrip("/") + "/v1/chat/completions",
                json={
                    "model": LITELLM_MODEL,
                    "messages": [
                        {"role": "system", "content": self._LLM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 10,
                },
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}",
                         "Content-Type": "application/json"},
                timeout=LLM_INTENT_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip().lower()
            # Normalize: map synonyms
            for intent in [INTENT_GREETING, INTENT_SYSTEM_INFO, INTENT_SEARCH,
                           INTENT_CLARIFY, INTENT_FALLBACK]:
                if intent in result:
                    return intent
            return None
        except Exception as e:
            logger.debug(f"LLM classification failed: {e}")
            return None

    def classify(self, query: str) -> str:
        """
        Classify user intent.

        Flow: rules → LLM → default "search"

        Returns one of: greeting, system_info, search, clarify, fallback
        """
        # Step 1: Rule-based (fast, <1ms)
        t0 = time.perf_counter()
        result = self._rule_classify(query)
        rule_time = (time.perf_counter() - t0) * 1000

        if result:
            logger.debug(f"Intent '{result}' via rules ({rule_time:.1f}ms)")
            return result

        # Step 2: LLM fallback (<2s)
        t0 = time.perf_counter()
        result = self._llm_classify(query)
        llm_time = (time.perf_counter() - t0) * 1000

        if result:
            logger.debug(f"Intent '{result}' via LLM ({llm_time:.1f}ms)")
            return result

        # Step 3: Safe default — treat as search
        logger.debug(f"Intent 'search' (default, rules={rule_time:.1f}ms, llm={llm_time:.1f}ms)")
        return INTENT_SEARCH


class Agent:
    """
    Agentic intelligence layer wrapping Searcher.

    Provides:
        - Intent classification (rule + LLM fallback)
        - Intent routing (static replies, search, clarification)
        - Multi-turn context memory (last N turns)
        - Graceful degradation when LLM is unavailable
        - Static responses for greeting and system_info (no LLM dependency)

    The agent does NOT modify search.py — it wraps Searcher.search().
    """

    # Static responses (no LLM dependency — always available)
    _GREETING_RESPONSE = (
        "Hello! 👋 I'm **CorpChat Intelligence** — your corporate chat analytics assistant. "
        "I can search across corporate WeCom conversations to find the information you need. "
        "Try asking about logistics quotes, investment opportunities, scam messages, or any topic in the conversations."
    )

    _SYSTEM_INFO_RESPONSE = (
        "I'm **CorpChat Intelligence** — an AI-powered search assistant for corporate chat messages.\n\n"
        "**Capabilities:**\n"
        "- Semantic hybrid search (BM25 + vector embeddings)\n"
        "- LLM query expansion for better recall\n"
        "- Graph-enhanced retrieval (traverses conversation relationships)\n"
        "- Cross-encoder reranking for result relevance\n\n"
        "**Data scope:**\n"
        "- 35 conversation threads across 30 contacts\n"
        "- Topics: business inquiries, quotations, investments, logistics, tech support,\n"
        "  invoices, contracts, quality issues, scam/phishing detection\n"
        "- Bilingual (Chinese + English) messages\n\n"
        "**Limitations:**\n"
        "- Can only search within the indexed message corpus\n"
        "- Cannot access external data or real-time feeds\n"
        "- LLM-dependent features (query expansion, agentic mode) degrade gracefully\n"
        "  when the LLM endpoint is unavailable\n\n"
        "How can I help you today?"
    )

    def __init__(
        self,
        searcher: Optional[Searcher] = None,
        classifier: Optional[IntentClassifier] = None,
        max_history: int = 10,
    ):
        """
        Args:
            searcher: A pre-constructed Searcher instance. If None, must call
                        set_searcher() before processing search queries.
            classifier: Intent classifier. If None, creates a default one.
            max_history: Maximum number of turns to keep in context memory.
        """
        self.searcher = searcher
        self.classifier = classifier or IntentClassifier()
        self.max_history = max_history
        self.chat_history: List[Dict[str, Any]] = []

    def set_searcher(self, searcher: Searcher):
        """Set or replace the Searcher instance (lazy loading support)."""
        self.searcher = searcher

    def _add_to_history(self, user_msg: str, bot_msg: str):
        """Add a turn to the multi-turn context memory."""
        self.chat_history.append({"user": user_msg, "bot": bot_msg})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def _get_context(self, query: str) -> str:
        """
        Build context from chat history + current query.
        For clarification queries, include the last turn for context.
        """
        if not self.chat_history:
            return query

        # Include last 3 turns as context
        recent = self.chat_history[-3:]
        context_parts = [f"Previous conversation:"]
        for turn in recent:
            context_parts.append(f"  User: {turn['user']}")
            context_parts.append(f"  Assistant: {turn['bot'][:100]}...")
        context_parts.append(f"Current query: {query}")
        return "\n".join(context_parts)

    def process(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        expand: bool = True,
        graph_expand: int = 1,
        label_filter: Optional[str] = None,
        search_mode: str = "hybrid",
    ) -> Tuple[str, str, List[Dict]]:
        """
        Process a user query through the agentic pipeline.

        Flow:
            1. Classify intent (rule-based → LLM fallback → default search)
            2. Route to handler based on intent
            3. For search intent: call Searcher.search() with params
            4. Return (intent, response, search_results)

        Args:
            query: User's input string.
            top_k: Number of search results to return (for search intent).
            use_rerank: Whether to use cross-encoder reranking.
            expand: Whether to use LLM query expansion.
            graph_expand: Number of graph expansion hops.
            label_filter: Optional label to filter results by.
            search_mode: "hybrid", "keyword", or "semantic".

        Returns:
            Tuple of (intent, response_text, search_results)
            - intent: the classified intent string
            - response_text: text to show the user (static reply or LLM answer)
            - search_results: list of result dicts (empty for non-search intents)
        """
        t0 = time.perf_counter()

        # Step 1: Classify intent
        intent = self.classifier.classify(query)
        classify_time = (time.perf_counter() - t0) * 1000

        # Step 2: Route based on intent
        if intent == INTENT_GREETING:
            self._add_to_history(query, self._GREETING_RESPONSE)
            return intent, self._GREETING_RESPONSE, []

        elif intent == INTENT_SYSTEM_INFO:
            self._add_to_history(query, self._SYSTEM_INFO_RESPONSE)
            return intent, self._SYSTEM_INFO_RESPONSE, []

        elif intent == INTENT_CLARIFY:
            response = "I'd be happy to clarify! Could you rephrase your question or provide more specific details about what you're looking for?"
            self._add_to_history(query, response)
            return intent, response, []

        elif intent == INTENT_FALLBACK:
            # Fallback → search (safe default per design)
            # Fall through to search with a note
            intent = INTENT_SEARCH

        # Step 3: Search intent (covers INTENT_SEARCH and INTENT_FALLBACK)
        if self.searcher is None:
            response = "Search system is not initialized. Please load the index first."
            self._add_to_history(query, response)
            return intent, response, []

        try:
            results = self.searcher.search(
                query,
                mode=search_mode,
                limit=top_k,
                expand=expand,
                graph_expand=graph_expand,
                label_filter=label_filter,
                use_rerank=use_rerank,
            )

            # Build response from results
            if results:
                # Extract context for potential LLM answer
                context_parts = [r.get("text", "") for r in results[:top_k]]
                context = "\n---\n".join(context_parts)

                # Try LLM answer if available
                llm_ok = self.classifier._check_llm()
                if llm_ok and LITELLM_API_KEY:
                    answer = self._generate_answer(query, context)
                    if answer is None:
                        # LLM failed — fall back to formatted results
                        answer = self._format_results_as_answer(query, results)
                else:
                    # Fallback: show top results
                    answer = self._format_results_as_answer(query, results)
            else:
                answer = "I couldn't find any relevant messages in the conversation corpus."

            routing_time = (time.perf_counter() - t0) * 1000
            logger.debug(
                f"Agent processed '{query[:30]}...' → intent={intent}, "
                f"{len(results)} results, total={routing_time:.1f}ms"
            )

            self._add_to_history(query, answer)
            return intent, answer, results

        except Exception as e:
            logger.error(f"Search failed in agent: {e}")
            response = f"Search encountered an error: {e}. Showing keyword-based results."
            # Retry with keyword mode if hybrid fails
            try:
                results = self.searcher.search(
                    query, mode="keyword", limit=top_k,
                    expand=False, graph_expand=0,
                    label_filter=label_filter, use_rerank=False,
                )
                if results:
                    answer = self._format_results_as_answer(query, results)
                    self._add_to_history(query, answer)
                    return INTENT_SEARCH, answer, results
            except Exception:
                pass

            self._add_to_history(query, response)
            return intent, response, []

    def _generate_answer(self, query: str, context: str) -> Optional[str]:
        """Generate LLM answer (requires LiteLLM)."""
        try:
            import requests
            resp = requests.post(
                LITELLM_BASE_URL.rstrip("/") + "/v1/chat/completions",
                json={
                    "model": LITELLM_MODEL,
                    "messages": [
                        {"role": "system", "content": (
                            "You are a helpful assistant answering questions based on retrieved chat messages. "
                            "Answer concisely in the same language as the query. "
                            "If the context doesn't contain the answer, say so."
                        )},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}",
                         "Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            return None

    def _format_results_as_answer(self, query: str, results: List[Dict]) -> str:
        """Format search results as a readable answer (fallback when LLM is down)."""
        if not results:
            return "No relevant messages found."

        parts = [f"Found {len(results)} relevant messages:\n"]
        for i, r in enumerate(results[:5], 1):
            meta = r.get("metadata", {})
            score = f"{r.get('score', 0):.4f}" if r.get("score") else "N/A"
            label = meta.get("label", "unknown")
            sender = meta.get("customer_name", meta.get("external_userid", "?"))
            text = r.get("text", "")[:200]
            parts.append(f"\n{i}. [{label}] {sender} → {text}")
        return "\n".join(parts)

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []


# ── Convenience: lazy-load agent with index ────────────────────────────────
_agent_instance: Optional[Agent] = None
_index_loaded: Optional[Any] = None


def load_agent(index_path: Optional[str] = None) -> Agent:
    """
    Load (or reuse cached) Agent with the search index.

    Uses module-level caching so repeated calls are fast.
    """
    global _agent_instance

    if _agent_instance is not None:
        return _agent_instance

    global _index_loaded
    if _index_loaded is None:
        _index_loaded = load_index(index_path or DEFAULT_INDEX_PATH)

    searcher = Searcher(_index_loaded)
    _agent_instance = Agent(searcher=searcher)
    return _agent_instance


def get_or_create_agent(searcher: Optional[Searcher] = None) -> Agent:
    """Get a fresh Agent (useful in tests for isolation)."""
    classifier = IntentClassifier()
    return Agent(searcher=searcher, classifier=classifier)
