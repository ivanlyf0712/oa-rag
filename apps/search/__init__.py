"""Contract search package - public API.

The full retrieval stack (index build, hybrid search, expansion, rerank,
agentic decision) lives in apps/search/_core.py. The unified contract search
service lives in apps/search/service.py. The CLI entrypoint lives in
apps/search_cli.py. Agentic routing is implemented in apps/search/agent.py.
"""

from apps.search.service import (  # noqa: F401
    OBSERVATION_ROW_BUDGET,
    RANK_RELEVANCE,
    RANK_RISK,
    SEMANTIC_CONTRACT_LIMIT,
    ContractSearchService,
    UnifiedQueryPlanner,
    format_contract_observation,
    format_contract_results,
)
from apps.search.result_store import (  # noqa: F401
    clear_results,
    snapshot_results,
    stash_results,
)
from apps.search._core import (  # noqa: F401
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_INDEX_PATH,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RERANKER_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
    LLM_KEYWORD_QUERY_WEIGHT,
    LLM_SEMANTIC_QUERY_WEIGHT,
    MAX_SEARCH_LIMIT,
    ORIGINAL_QUERY_WEIGHT,
    RRF_K_VALUE,
    AgenticDecider,
    IndexBuilder,
    QueryExpander,
    Reranker,
    Searcher,
    load_index,
)

from apps.search.agent import CrossTableAgent  # noqa: F401
from apps.search.intents import (  # noqa: F401
    INTENT_CLARIFY,
    INTENT_COUNTERPARTY,
    INTENT_GENERAL,
    INTENT_RENEWAL,
    INTENT_RISK,
    TOOL_CONTRACT_SEARCH,
    TOOL_RISK_SEARCH,
    TOOL_NONE,
    VALID_INTENTS,
)
from apps.search.router import SearchRouter  # noqa: F401
from apps.search.langchain_agent import (  # noqa: F401
    AgentConfigError,
    LangChainAgent,
    build_default_llm,
    build_langchain_tools,
)
