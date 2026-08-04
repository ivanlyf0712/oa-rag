# CorpChat Agentic Intelligence Layer — Implementation Report

## 1. Architecture Design

### Approach Chosen: Independent Module (`agent.py`) Wrapping `Searcher`

**Decision:** Build the agentic layer as a **separate module** (`apps/corpchat/agent.py`) that wraps `Searcher.search()` — **no modifications to `search.py`**.

### Why this approach?

| Criterion | Rationale |
|---|---|
| **Minimal search.py changes** | The agent calls `Searcher.search()` as-is. All search pipeline logic (BM25, vector, RRF, graph, reranker) remains untouched. |
| **LLM unavailable now** | Rule-based intent classification works without LLM. When LLM is down, the agent defaults unknown queries to "search". Static responses for greeting/system_info require no LLM. |
| **Performance** | Rule-based classification: <1ms. Greeting routing: <10ms. Total agent overhead (excluding search): <500ms. |
| **Testability** | Tests can inject mock searchers and mock classifiers independently. 27 tests run in ~13s with no LLM calls. |
| **Reusability** | The agent can be used from CLI, Streamlit UI, or any external caller. |

### Why not integrate into search.py?

- `search.py` is the **retrieval engine** — adding conversational routing violates single-responsibility.
- `search.py` already has `AgenticDecider` for parameter tuning (expand/graph/rerank decisions), not for conversational routing.
- Keeping them separate follows the **open/closed principle**: search.py is closed for modification, agent.py extends it.

### Trade-offs

| Pro | Con |
|---|---|
| Zero search.py modifications | Two classes handle "agency" (AgenticDecider in search.py + Agent in agent.py) |
| Works without LLM | Rule-based classification less accurate than LLM for edge cases |
| <1ms rule classification | Requires manual keyword maintenance |

## 2. Complete Code

### Module: `apps/corpchat/agent.py`

Two main classes:

#### `IntentClassifier`
- **Rule-based first** (keyword matching, <1ms)
- **LLM fallback** (LiteLLM classification, 2s timeout)
- **Default to "search"** when LLM unavailable (safe degradation)

Classification flow:
```
User query → Rule matching (keywords) → hit? → yes → return category
                                ↓ no
                        LLM classification → 2s timeout → hit? → yes → return category
                                                ↓ no/timeout
                                        default → "search"
```

5 intent categories: `greeting`, `system_info`, `search`, `clarify`, `fallback`

#### `Agent`
- Routes intents to handlers
- **Static responses** for `greeting` and `system_info` (pre-written, no LLM dependency)
- **Search intent** wraps `Searcher.search()` with enhancement parameters
- **Multi-turn context memory** (configurable, last N turns)
- **Graceful degradation**: when LLM is down, formats results as readable text instead of LLM answer; on search error, retries with keyword-only mode

### Key Design Decision: System_info checked before Greeting

In `_rule_classify()`, system_info keywords are checked **before** greeting keywords. This prevents false matches: "what can you do" contains "yo" (inside "you"), which would otherwise match the greeting keyword "yo" and return `greeting` instead of `system_info`.

```python
# System info checked FIRST (before greeting)
for kw in SYSTEM_INFO_KEYWORDS:
    if kw in q_lower:
        return INTENT_SYSTEM_INFO

# Greeting checked second (short queries only)
if len(q_lower) <= 15:
    for kw in GREETING_KEYWORDS:
        if kw in q_lower:
            return INTENT_GREETING
```

## 3. Test Report (27 tests, all passing)

### Classification Tests (8 tests)

| # | Test | Input | Expected | Actual | Pass |
|---|------|-------|----------|--------|------|
| 1 | `test_greeting_en` | "Hi", "hello", "hey" | `greeting` | `greeting` | ✅ |
| 2 | `test_greeting_zh` | "你好", "嗨", "哈囉" | `greeting` | `greeting` | ✅ |
| 3 | `test_system_info_en` | "who are you", "what can you do" | `system_info` | `system_info` | ✅ |
| 4 | `test_system_info_zh` | "你是誰", "你能做什麼" | `system_info` | `system_info` | ✅ |
| 5 | `test_search_explicit` | "找物流報價 方案", "搜尋投資" | `search` | `search` | ✅ |
| 6 | `test_fallback_to_search` | "xyzzy foobar quux" | `search` (default) | `search` | ✅ |
| 7 | `test_clarify` | "能再說詳細一些嗎" | `clarify` | `clarify` | ✅ |
| 8 | `test_rule_speed` | 100 queries | <1ms avg | <1ms | ✅ |

### Routing Tests (6 tests)

| # | Test | Input | Expected | Actual | Pass |
|---|------|-------|----------|--------|------|
| 9 | `test_greeting_routes_static` | "Hi" | greeting + static reply | ✅ | ✅ |
| 10 | `test_system_info_routes_static` | "you are who" | system_info + capability desc | ✅ | ✅ |
| 11 | `test_search_routes_to_searcher` | "物流報價 方案" | search + results | ✅ | ✅ |
| 12 | `test_clarify_routes_to_clarify` | "能再說詳細一些嗎" | clarify + ask to rephrase | ✅ | ✅ |
| 13 | `test_fallback_routes_to_search` | "xyzzy ..." | search (safe default) | ✅ | ✅ |
| 14 | `test_no_searcher_returns_error` | "找物流" (no searcher) | error message | ✅ | ✅ |

### Graceful Degradation Tests (4 tests)

| # | Test | Scenario | Expected | Actual | Pass |
|---|------|----------|----------|--------|------|
| 15 | `test_llm_down_search_still_works` | LLM down, search query | results returned | 15 results | ✅ |
| 16 | `test_llm_down_no_crash_on_greeting` | LLM down, greeting | static reply | ✅ | ✅ |
| 17 | `test_llm_down_no_crash_on_system_info` | LLM down, system query | static reply | ✅ | ✅ |
| 18 | `test_llm_down_no_results_message` | Mock empty results | helpful "couldn't find" message | ✅ | ✅ |

### Multi-turn Context Tests (3 tests)

| # | Test | Action | Expected | Actual | Pass |
|---|------|--------|----------|--------|------|
| 19 | `test_history_grows` | 2 queries | 2 history entries | ✅ | ✅ |
| 20 | `test_history_max` | 5 queries (max=2) | 2 entries (capped) | ✅ | ✅ |
| 21 | `test_reset_clears_history` | reset() after 2 queries | 0 entries | ✅ | ✅ |

### Search Quality Tests (3 tests)

| # | Test | Query | Expected | Actual | Pass |
|---|------|-------|----------|--------|------|
| 22 | `test_logistics_query` | "物流報價 方案" | logistics in top results | ✅ | ✅ |
| 23 | `test_investment_query` | "投資美國債券跟藍籌股" | investment message in results | ✅ | ✅ |
| 24 | `test_label_filter_through_agent` | "物流" + label_filter | all results have that label | ✅ | ✅ |

### Performance Tests (3 tests)

| # | Test | Metric | Constraint | Actual | Pass |
|---|------|--------|------------|--------|------|
| 25 | `test_intent_classification_under_1ms` | Classify 700 queries | <1ms avg | <1ms | ✅ |
| 26 | `test_agent_overhead_excludes_search` | Classify 100 "Hi" | <500ms | <1ms | ✅ |
| 27 | `test_greeting_response_time` | Full greeting route | <10ms | <10ms | ✅ |

### Summary

```
============================= 27 passed in 13.35s ==============================
```

**Test command:** `/Users/ivanlee/miniconda3/envs/ocr/bin/python -m pytest tests/test_agent.py -v`

## 4. Integration Guide (Streamlit)

### Option A: Replace existing search with Agent

In `app.py`, the chat flow on the Search page can route queries through the agent instead of calling `_run_search` directly:

```python
from apps.corpchat.agent import load_agent

# In the Search page section:
agent = load_agent()

if pending_turn:
    query = pending_turn["query"]
    intent, response, search_results = agent.process(
        query,
        top_k=top_k,
        use_rerank=use_rerank,
        expand=expand,
        graph_expand=graph_expand,
        label_filter=label_filter or None,
    )
    # Display intent for debugging
    st.caption(f"Intent: {intent}")
    # Display the agent's response
    st.markdown(response)
```

### Option B: Add an "Agent Mode" switch

Add a checkbox in the Enhancements panel:

```python
with st.expander("Enhancements", expanded=not st.session_state.searching):
    agent_mode = st.checkbox("Agent mode", value=False,
                              help="Route through intent classifier",
                              disabled=st.session_state.searching)
    use_rerank = st.checkbox("Reranker", value=True, ...)
    # ...

# In search handler:
if agent_mode:
    intent, response, results = agent.process(
        query,
        top_k=top_k,
        use_rerank=use_rerank,
        expand=expand,
        graph_expand=graph_expand,
        label_filter=label_filter or None,
    )
else:
    results, raw_hits = _run_search(query, top_k, use_rerank, expand, ...)
```

### Option C: Programmatic API

```python
from apps.corpchat.agent import Agent, IntentClassifier
from apps.corpchat.search import Searcher, load_index

# Load index and create searcher
embeddings = load_index()
searcher = Searcher(embeddings)

# Create agent
agent = Agent(searcher=searcher, classifier=IntentClassifier())

# Process a query — returns (intent, response_text, search_results)
intent, response, results = agent.process(
    "找物流報價",
    top_k=5,
    use_rerank=True,
    expand=True,
    graph_expand=1,
)

print(f"Intent: {intent}")
print(f"Response: {response[:200]}...")
print(f"Results: {len(results)} hits")
```

## 5. Improvement Suggestions (3 Directions)

### 5.1 Contextual Disambiguation (Multi-turn LLM Classification)

Currently, the LLM classifier gets only the current query. When LLM is available, include the last 3 turns of chat history as context so the classifier can resolve ambiguous queries like "Show me more" or "What about the other one?" — queries that reference previous results.

**Implementation:** Pass `_get_context()` output to the LLM classification prompt.

### 5.2 Hybrid Search Mode Selection

Currently, the agent always uses `search_mode="hybrid"`. When LLM is available, the agent could ask the LLM to determine whether a query is better served by keyword, semantic, or hybrid search — e.g., exact ID lookups → keyword, conceptual queries → semantic.

**Implementation:** Add an LLM prompt that classifies search type alongside intent.

### 5.3 Agent Memory (Persistent Context)

Currently, chat history is in-memory only and reset on page reload. A persistent conversation store (SQLite or in-memory keyed by session ID) would allow the agent to remember context across visits — e.g., "show me more from the logistics conversation we saw earlier."

**Implementation:** Add a `ConversationStore` class that serializes chat history to SQLite, keyed by session ID. The Agent constructor accepts an optional `session_id`.
