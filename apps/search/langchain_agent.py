"""LangChain tool-calling agent for unified contract search.

Wraps the unified contract retrieval pipeline as a LangChain tool and lets
the LLM select it via tool calling. All contract-domain questions — general
content, counterparty, renewal/expiry, and risk/compliance filters — route
through the single contract_search tool.

  - contract_search - the default tool for all contract-domain questions.

Design notes
------------
* No retrieval rewrite. The agent is a thin adapter over the existing
  contract_tool callable (built in apps/search_cli.py),
  so all retrieval behaviour underneath the agent is preserved.
* Provider-agnostic LangChain interface. Any chat model implementing the
  LangChain BaseChatModel tool-calling interface can be injected (a fake in
  tests, ...). The default factory builds a LiteLLM-backed model from
  LITELLM_* environment variables (see apps/search/litellm_client.py).
* Single LLM provider. The LiteLLM proxy is the one and only model source;
  there is no secondary-model failover chain.
* Safe degradation. If the model is unavailable or the ReAct loop fails, the
  agent falls back to the deterministic SearchRouter policy: contract search
  stays the default and a vague query triggers a clarification question.
  Tool failures never crash the caller.
* UI-independent. The orchestration layer is importable and testable without
  Streamlit; the app only consumes the returned dict.

Public result contract (same public seam as CrossTableAgent)::

    {
      "output":     str,         # final natural-language answer
      "intent":     str,         # detected intent
      "tool":       str,         # tool that ran
      "tool_calls": List[Dict],  # executed calls (tool, tool_input, filters, observation)
      "steps":      List[Dict],  # process timeline (icon, label, detail)
      "success":    bool,
      "fallback":   bool,        # True when the deterministic (non-LLM) path was used
      "clarify":    bool,        # True when the agent is asking for clarification
      "observation": str,        # optional raw tool observation (evidence)
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from apps.search._core import LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL
from apps.search.hindsight_client import needs_recall, recall as hs_recall
from apps.search.litellm_client import LiteLLMClient
from apps.search.intents import (
    INTENT_CLARIFY,
    INTENT_GENERAL,
    INTENT_RISK,
    INTENT_TO_TOOL,
    TOOL_CONTRACT_SEARCH,
    TOOL_CONTRACTS_WHERE,
    TOOL_NONE,
    default_decision,
    infer_intent_from_query,
)
from apps.search.synthesis import AnswerSynthesizer
from apps.search.router import SearchRouter
from apps.search.service import _looks_like_ref_no

# Pydantic is a soft dependency — if missing we degrade to the bare Dict args_schema
# the agent already used, so the LLM sees no filter-key guidance (existing behaviour).
try:
    from pydantic import BaseModel, Field
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda *a, **kw: None

# Pull the actual label dictionaries from the DB layer so the LLM can map
# user language onto the real coded labels stored in the contract metadata.
try:
    from core.db import STATUS_LABELS as _DB_STATUS_LABELS, CONTRACT_TYPE_LABELS as _DB_CTYPE_LABELS
except ImportError:
    _DB_STATUS_LABELS: dict = {}
    _DB_CTYPE_LABELS: dict = {}

logger = logging.getLogger("oa-search.langchain-agent")


class AgentConfigError(RuntimeError):
    """Raised when the LangChain agent cannot be configured."""


# Primary provider: LiteLLM proxy (LITELLM_* env, default dseek-v4-flash).
ContractTool = Callable[[str, Dict[str, str]], str]
RiskTool = Callable[[str], str]

_DEFAULT_CLARIFICATION = (
    "Could you narrow down which contract, counterparty, or risk area you mean?"
)


def _missing_contract_tool(query: str, filters: Dict[str, str]) -> str:
    raise RuntimeError("contract_search tool is not configured")


from apps.search.status_labels import STATUS_ALIASES as _SHARED_STATUS_ALIASES
from apps.search.filters import (
    infer_contract_filters as _infer_filters,
    CONTRACT_TYPE_ALIASES as _FILTERS_CONTRACT_TYPE_ALIASES,
    DEPARTMENT_ALIASES as _FILTERS_DEPARTMENT_ALIASES,
)

# ── Label dictionaries the LLM sees in the tool args_schema description ──
# Format e.g. "Draft", "Pending Preliminary Review", ...
_STATUS_LIST = list(_DB_STATUS_LABELS.values())
# "NDA / Confidentiality Agreement", "MOU / LOI", ...
_CTYPE_LIST = list(_DB_CTYPE_LABELS.values())
_STATUS_DESC = ", ".join(f'"{v}"' for v in _STATUS_LIST) if _STATUS_LIST else '"Completed", "Draft", "Pending"'
_CTYPE_DESC = ", ".join(f'"{v}"' for v in _CTYPE_LIST) if _CTYPE_LIST else '"NDA", "Service Agreement", "Procurement", "Lease", "Others"'


class _ContractFiltersSchema(BaseModel):
    """Explicit tool args_schema so the LLM sees the valid filter keys & values."""
    query: str = Field(description="Natural-language query to search the contract corpus")
    contract_type: str = Field(
        default="",
        description=f"Contract type to filter by. Valid values: {_CTYPE_DESC}. "
                    "Leave empty when unspecified."
    )
    status: str = Field(
        default="",
        description=f"Status to filter by. Valid values: {_STATUS_DESC}. "
                    "Leave empty when unspecified."
    )
    counterparty_name: str = Field(
        default="",
        description="Counterparty company name to filter by. Leave empty when unspecified."
    )
    department: str = Field(
        default="",
        description="Department name to filter by. Leave empty when unspecified."
    )
    date_from: str = Field(default="", description="Earliest contract start date (YYYY-MM-DD)")
    date_to: str = Field(default="", description="Latest contract end date (YYYY-MM-DD)")
    expired: bool = Field(default=False, description="Whether to filter to expired contracts only")
    contract_id: str = Field(default="", description="Contract reference number to filter by")
    # Pre-built filters dict (legacy / scripted callers). Merged with the
    # individual fields above; individual fields win when both are present.
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional pre-built filters dict; merged with the fields above."
    )


# ── Greeting / small-talk fast-path keywords ────────────────────
_GREETING_PATTERNS = (
    "hello", "hi ", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "what's up", "sup",
    "how are you", "how do you do",
    "你好", "嗨", "早上好", "下午好", "晚上好",
)


# ─────────────────────────────────────────────────────────────────────
# LangChain tool adapters
# ─────────────────────────────────────────────────────────────────────
def build_langchain_tools(
    contract_tool: ContractTool,
    where_tool: Optional[Callable[[str], str]] = None,
) -> List[Any]:
    """Wrap the raw callables as LangChain @tool objects.

    Kept separate from the agent so the tools can be built/inspected/tested
    independently. Each tool returns the observation as a plain string so it
    can be surfaced in the UI as evidence. Candidate 2: there is no separate
    risk tool; risk is handled inside the unified contract_search service.
    """
    from langchain_core.tools import tool

    @tool(TOOL_CONTRACT_SEARCH,
          args_schema=_ContractFiltersSchema if _PYDANTIC_AVAILABLE else None)
    def contract_search(query: str, filters: Optional[Dict[str, Any]] = None,
                        # Accept individual filter kwargs when the LLM supplies them
                        # via args_schema; merged into the filters dict.
                        contract_type: str = "",
                        status: str = "",
                        counterparty_name: str = "",
                        department: str = "",
                        date_from: str = "",
                        date_to: str = "",
                        expired: bool = False,
                        contract_id: str = "",
                        ) -> str:
        """Search the contract corpus semantically (hybrid keyword/vector).

        Use for general contract questions: clauses, breach, termination,
        liability, amounts, counterparties, renewal/expiry, dates, and
        risk/compliance filters. This is the DEFAULT tool for any
        contract-content question.

        Filter fields (all optional; omit when the user did not specify):
        contract_type, status, counterparty_name, department, date_from,
        date_to, expired, contract_id.
        """
        merged = dict(filters or {})
        for key, val in [("contract_type", contract_type),
                         ("status", status),
                         ("counterparty_name", counterparty_name),
                         ("department", department),
                         ("date_from", date_from),
                         ("date_to", date_to),
                         ("expired", expired),
                         ("contract_id", contract_id)]:
            if val:
                # When the LLM passes the value via the individual kwarg
                # (from the Pydantic args_schema), prefer it over the dict.
                merged.setdefault(key, val)
        return contract_tool(query, merged)

    tools = [contract_search]

    if where_tool is not None:
        @tool(TOOL_CONTRACTS_WHERE)
        def contracts_where(condition: str) -> str:
            """Exact structured retrieval over the contracts DB (SQL filter,
            not semantic/vector). Use for 'list all contracts with <exact
            condition>' requests: amount comparisons (over HKM), date
            bounds (ending before 2027), coded flag labels (needs legal
            review, risk not accepted, external guarantees), or status.
            Bare 'list all' / empty condition returns every contract.
            'list all'-style queries with no semantic content route here,
            never to vector search.

            Args:
                condition: natural-language filter, e.g.
                    'contracts over HKM ending before 2027',
                    'contracts needing legal review', or '' for all contracts.
            """
            return where_tool(condition)

        tools.append(contracts_where)

    # Candidate 2: no separate risk_search tool. Risk/compliance screening is
    # handled by the unified contract_search tool (the service extracts risk
    # filters and ranks every candidate set by risk), so the LLM is always
    # offered a single search entry point.

    return tools


def build_default_llm(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    """Build the primary chat model from LITELLM_* config.

    Returns a _LiteLLMWrapper (a real BaseChatModel) against the
    OpenAI-compatible LiteLLM endpoint. Raises AgentConfigError when
    langchain-core is missing or the base URL is unconfigured so callers
    can transparently fall back to the manual router agent.

    NOTE: this used to return ChatOllama (local Ollama). Since the primary
    provider is now the LiteLLM proxy, ChatOllama would POST to the
    Ollama-native /api/chat path and get a 404 from the proxy.
    """
    if not _BASE_CHAT_MODEL_AVAILABLE:
        raise AgentConfigError(
            "langchain-core is not installed. Install with: pip install langchain-core"
        )
    base = (api_base or LITELLM_BASE_URL or "").strip().rstrip("/")
    if not base:
        raise AgentConfigError("LITELLM_BASE_URL is not configured.")
    return _LiteLLMWrapper(
        api_base=base,
        api_key=(api_key or LITELLM_API_KEY or "").strip(),
        model=(model or LITELLM_MODEL or "").strip(),
        temperature=temperature,
    )


def check_llm_health(deep: bool = False) -> Dict[str, Any]:
    """Check the LiteLLM provider (sole LLM source) and report the serving model.

    Phase 1 (always): fast GET /v1/models probe verifies the proxy is
    reachable and the API key is valid (~0.1s).

    Phase 2 (deep=True only): lightweight chat completion (1 token) verifies
    the proxy can actually generate. Catches 502/504/timeout that only show
    up during inference, but costs a full remote round-trip (0.6-20s+ on a
    loaded proxy), so it only runs on demand (sidebar "Recheck LLM" button),
    never on a Streamlit rerun critical path.

    Returns ok (bool), provider, model, base, and an
    error message string (empty means healthy).
    """
    client = LiteLLMClient()
    if not client.is_available():
        return {"ok": False, "provider": "none", "model": client.model,
                "base": client.api_base,
                "error": "LiteLLM proxy unreachable at %s" % client.api_base}
    if not deep:
        return {"ok": True, "provider": "litellm", "model": client.model,
                "base": client.api_base, "error": ""}
    # Phase 2: verify generation works (catches 504/timeout during inference).
    probe = client.chat([{"role": "user", "content": "ping"}],
                        max_tokens=1, temperature=0, timeout=10)
    if not probe:
        return {"ok": False, "provider": "litellm", "model": client.model,
                "base": client.api_base,
                "error": ("LiteLLM proxy reachable but chat generation fails "
                          "(502/504/timeout). The proxy may be overloaded or "
                          "the upstream model is unavailable.")}
    return {"ok": True, "provider": "litellm", "model": client.model,
            "base": client.api_base, "error": ""}


# ── LangGraph availability ─────────────────────────────────────
# LangGraph V1.0 moved create_react_agent to langchain.agents.create_agent;
# prefer that supported location and only fall back to the legacy
# langgraph.prebuilt symbol when the full langchain package is not installed.
# The legacy fallback emits a LangGraphDeprecatedSinceV10 warning; we silence
# that single known warning at the import site so it does not drown out real
# warnings in the test suite (the fallback is still the only available symbol
# in a langgraph-only environment).
import warnings as _warnings

try:
    from langchain.agents import create_agent as _create_react_agent
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # known legacy-path deprecation
            from langgraph.prebuilt import create_react_agent as _create_react_agent
        _LANGGRAPH_AVAILABLE = True
    except ImportError:
        _create_react_agent = None
        _LANGGRAPH_AVAILABLE = False


# ── BaseChatModel stubs (langchain-core optional) ──────────────────
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    _BASE_CHAT_MODEL_AVAILABLE = True
except ImportError:
    _BASE_CHAT_MODEL_AVAILABLE = False
    AIMessage = HumanMessage = SystemMessage = object
    BaseChatModel = object
    ChatGeneration = ChatResult = object


class _LiteLLMWrapper(BaseChatModel):
    """Minimal BaseChatModel around LiteLLMClient (native tool calling).

    All HTTP/transport concerns live in LiteLLMClient; this class only
    adapts LangChain message/tool formats and the BaseChatModel protocol.
    """

    def __init__(self, api_base: str, api_key: str, model: str,
                 temperature: float = 0.0, **kwargs):
        if not _BASE_CHAT_MODEL_AVAILABLE:
            raise RuntimeError(
                "langchain-core is not installed. Install with: pip install langchain-core"
            )
        super().__init__(**kwargs)
        self._api_base = (api_base or "").rstrip("/")  # avoid //chat/completions
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._client: Optional[LiteLLMClient] = None
        self._bound_tools: List[Any] = []

    def _get_client(self) -> LiteLLMClient:
        if self._client is None:
            self._client = LiteLLMClient(
                api_base=self._api_base, api_key=self._api_key, model=self._model)
        return self._client

    @property
    def _llm_type(self) -> str:
        return "litellm-wrapper"

    @property
    def _identifying_params(self):
        return {"model": self._model}

    def bind_tools(self, tools, **kwargs):
        """Store tool schemas so _generate() can send them to the LLM."""
        self._bound_tools = list(tools)
        return self

    def _messages_to_openai(self, messages) -> List[Dict]:
        """Convert LangChain messages to OpenAI chat-format dicts."""
        raw: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                raw.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                raw.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc.get("id", ""), "type": "function",
                         "function": {"name": tc.get("name", ""),
                                      "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False)}}
                        for tc in tool_calls]
                raw.append(entry)
            elif getattr(msg, "type", "") == "tool":
                raw.append({"role": "tool", "tool_call_id": getattr(msg, "tool_call_id", ""),
                            "content": str(msg.content)})
            else:
                raw.append({"role": "user", "content": str(getattr(msg, "content", ""))})
        return raw

    @staticmethod
    def _tools_to_schema(tools) -> List[Dict]:
        """Convert LangChain tools to OpenAI function-calling schema."""
        schema = []
        for tool in tools:
            try:
                name = tool.name if hasattr(tool, "name") else getattr(tool, "func", tool).__name__
            except Exception:
                name = str(tool)
            try:
                args_schema = tool.args_schema
            except Exception:
                args_schema = None
            if args_schema is not None:
                try:
                    parameters = args_schema.model_json_schema()
                except Exception:
                    try:
                        parameters = args_schema.schema()
                    except Exception:
                        parameters = {"type": "object", "properties": {}}
            else:
                parameters = {"type": "object", "properties": {}}
            schema.append({"type": "function",
                           "function": {"name": name,
                                        "description": tool.description if hasattr(tool, "description") else "",
                                        "parameters": parameters}})
        return schema

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Call the LLM with real tool calling; return ChatResult.

        When tools are bound, the schema is sent with the request. If the
        model responds with tool_calls, they are parsed and returned as
        AIMessage.tool_calls (LangGraph executes them and calls back with the
        results). Otherwise the model's content is the final answer.
        """
        client = self._get_client()
        raw_messages = self._messages_to_openai(messages)
        tools_schema = self._tools_to_schema(self._bound_tools) if self._bound_tools else None

        result = client.chat_message(
            raw_messages,
            tools=tools_schema,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens"),
            timeout=kwargs.get("timeout"),
        )
        if result is None:
            # LLM unreachable → empty AIMessage ends the loop gracefully;
            # LangChainAgent.process() then falls back to the router.
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        content = result.get("content") or ""
        raw_tool_calls = result.get("tool_calls") or []
        if raw_tool_calls:
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError) as e:
                    # 非法 tool_calls → 抛异常, 让 LangGraph 重试本轮。
                    raise ValueError(
                        f"LLM returned malformed tool arguments for {name!r}: {args_raw!r} ({e})"
                    )
                parsed_tool_calls.append({
                    "name": name,
                    "args": args if isinstance(args, dict) else {},
                    "id": tc.get("id", ""),
                })
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="", tool_calls=parsed_tool_calls))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content or ""))])

    # NOTE: no custom invoke() override — the BaseChatModel.invoke(
    #   input, config=None, *, stop=None, **kwargs) implementation is
    #   inherited. LangGraph passes config positionally, so an override
    #   with a narrower signature breaks the ReAct loop.


# ── Language detection ──────────────────────────────────────────────
def _detect_language(user_input: str) -> str:
    trad_chars = "維認體機關係臺灣門東車馬為與從來個們說時後這那裡會對過還進動開點學問發約訊郵簽話電號碼員務"
    simp_chars = "维认体机关系台湾门东车马为与从来个们说时候这里会对过还进动开点学问发约讯邮签话电号码员务"
    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in user_input)
    if not has_cjk:
        return "en"
    trad_score = sum(1 for ch in user_input if ch in trad_chars)
    simp_score = sum(1 for ch in user_input if ch in simp_chars)
    return "zh-TW" if trad_score > simp_score else "zh-CN"


# ── Greeting fast-path ──────────────────────────────────────────────
def _quick_respond(user_input: str) -> Optional[str]:
    q = user_input.lower().strip()
    for pattern in _GREETING_PATTERNS:
        if pattern in q or q.startswith(pattern):
            lang = _detect_language(user_input)
            if lang == "en":
                return ("Hello! I'm your contract search assistant. I can help you "
                        "search contracts and identify risk/compliance flags. How can I help you?")
            return ("你好！我是合同搜索助手。我可以帮你搜索合同内容和识别风险/合规标记。"
                    "请问有什么可以帮你的？")
    return None


# ── Notifying tool wrapper ──────────────────────────────────────────
def _notifying_tool(tool, on_tool_callback=None, tool_meta_log=None):
    import functools
    from langchain_core.tools import StructuredTool
    func = tool.func
    @functools.wraps(func)
    def _run(*args, **kwargs):
        if on_tool_callback is not None:
            try:
                on_tool_callback(tool.name, dict(kwargs))
            except Exception:
                pass
        result = func(*args, **kwargs)
        if tool_meta_log is not None:
            try:
                tool_meta_log.append({"tool": tool.name, "args": dict(kwargs)})
            except Exception:
                pass
        return result
    return StructuredTool.from_function(func=_run, name=tool.name, description=tool.description,
                                        args_schema=tool.args_schema)

class LangChainAgent:
    """LangChain tool-calling agent over the unified contract search pipeline.

    Candidate 2: risk/compliance screening is handled inside the unified
    contract_search service, so there is no separate risk tool.

    Parameters
    ----------
    contract_tool:
        The existing contract retrieval callable (see apps/search_cli.py builder).
    risk_tool:
        Deprecated/ignored. Accepted for backward compatibility; risk queries
        route to the unified contract_search tool.
    llm:
        Any LangChain chat model. If omitted, build_default_llm() is built
        lazily on the first process() call.
    router:
        Fallback SearchRouter used when the LLM path is unavailable.
    synthesize:
        Optional callable (query, tool, observation) -> str producing the
        final answer. Defaults to a deterministic summariser so the agent is
        fully testable without a live LLM.
    profile:
        Optional DispositionProfile persona to inject into the system prompt.
    """

    def __init__(
        self,
        contract_tool: Optional[ContractTool] = None,
        risk_tool: Optional[RiskTool] = None,
        where_tool: Optional[Callable[[str], str]] = None,
        llm: Any = None,
        router: Optional[SearchRouter] = None,
        synthesize: Optional[Callable[[str, str, str], str]] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        profile: Any = None,
        hindsight_bank: Optional[str] = None,
    ):
        self.contract_tool: ContractTool = contract_tool or _missing_contract_tool
        # Candidate 2: risk queries route to the unified contract_search tool;
        # risk scoring/ranking happens inside the search service. The risk_tool
        # parameter is accepted for backward compatibility but ignored.
        self._llm = llm
        self._api_base = api_base
        self._model = model
        self.profile = profile
        # Hindsight 记忆银行 ID; None/空 → 不做跨会话记忆 recall
        self.hindsight_bank = hindsight_bank or os.getenv("HINDSIGHT_BANK_ID") or None
        self.router = router or SearchRouter(api_base=api_base, model=model or LITELLM_MODEL)
        self._synthesize = synthesize or self._default_synthesize
        self._has_where_tool = where_tool is not None
        self._tools = build_langchain_tools(
            self.contract_tool, where_tool=where_tool)
        self._tool_map = {t.name: t for t in self._tools}
        self._steps: List[Dict[str, Any]] = []
        # LangGraph ReAct agent (lazy-initialized)
        self._react_agent: Optional[Any] = None
        # Per-request tool-execution callback (UI progress)
        self._on_tool_callback: Optional[Callable] = None
        # Per-request tool meta log (tool name + args in order)
        self._tool_meta_log: List[Dict[str, Any]] = []
        # Synthesis deep module (Candidate 3), built lazily around the LLM.
        self._synthesizer: Optional[AnswerSynthesizer] = None

    def _add_step(self, icon: str, label: str, detail: str = "") -> None:
        self._steps.append({"icon": icon, "label": label, "detail": detail})

    def _get_synthesizer(self) -> AnswerSynthesizer:
        """Return the shared AnswerSynthesizer, rebuilding if the LLM changed.

        The LLM is resolved lazily (it may be unavailable on first use), so the
        synthesizer is rebuilt whenever the resolved LLM object changes.
        """
        llm = self._get_llm()
        if self._synthesizer is None or self._synthesizer._llm is not llm:
            self._synthesizer = AnswerSynthesizer(llm=llm)
        return self._synthesizer

    def _get_llm(self) -> Optional[Any]:
        """Return the shared chat model (_LiteLLMWrapper), or None when unavailable.

        Single provider (LiteLLM proxy): there is no secondary model; a None
        return routes callers to the deterministic, non-LLM paths.
        """
        if self._llm is not None:
            return self._llm
        try:
            self._llm = build_default_llm(api_base=self._api_base, model=self._model)
        except Exception as e:
            logger.warning("LLM unavailable: %s", e)
            self._add_step("warn", "Model unavailable", str(e)[:100])
            return None
        return self._llm

    def _result(self, *, output, intent, tool, tool_calls, success, fallback,
                clarify, observation) -> Dict[str, Any]:
        return {
            "output": output,
            "intent": intent,
            "tool": tool,
            "tool_calls": tool_calls,
            "steps": self._steps,
            "success": success,
            "fallback": fallback,
            "clarify": clarify,
            "observation": observation,
        }

    def _default_synthesize(self, query: str, tool: str, observation: str) -> str:
        """Produce a concise overall summary of the retrieved evidence.

        The synthesis policy (prompt wording, LLM plumbing, deterministic
        no-LLM fallback) lives in the shared ``apps.search.synthesis`` deep
        module (Candidate 3); this method only resolves the LLM and delegates
        so the seam stays stable for existing callers/tests.
        """
        return self._get_synthesizer().synthesize(query, tool, observation)

    @staticmethod
    def _fallback_summary(observation: str) -> str:
        """Deterministic no-LLM summary (delegates to the synthesis module)."""
        return AnswerSynthesizer.fallback_summary(observation)

    # ── main entry ─────────────────────────────────────────────
    def process(self, user_input: str, on_stage: Optional[Callable] = None,
                on_tool: Optional[Callable] = None,
                history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Route user_input via LangChain tool calling and answer it.

        Args:
            user_input: The user query.
            on_stage: Optional callback for stage updates (label, detail).
            on_tool: Optional callback for tool-execution updates (tool_name, args).
            history: Optional list of previous turn dicts [{"role": ..., "content": ...}]
                     for multi-turn context.
        """

        def _stage(label: str, detail: str = ""):
            if on_stage:
                try:
                    on_stage(label, detail)
                except Exception:
                    pass

        self._steps = []
        self._on_tool_callback = on_tool
        self._tool_meta_log.clear()  # in-place: cached tool closures hold this list
        query = (user_input or "").strip()
        if not query:
            return self._result(
                output="Please enter a contract search query.",
                intent=INTENT_CLARIFY, tool=TOOL_NONE, tool_calls=[],
                success=False, fallback=False, clarify=True, observation="",
            )

        # Step 0: Greeting / small-talk fast-path
        quick = _quick_respond(query)
        if quick is not None:
            self._add_step("👋", "Greeting", "fast-path response")
            return self._result(
                output=quick, intent=INTENT_GENERAL, tool=TOOL_NONE, tool_calls=[],
                success=True, fallback=False, clarify=False, observation="",
            )

        # Step 1: deterministic fast-route. Keyword-confident queries (risk
        # phrases, bare ref numbers) skip the ReAct decision call entirely,
        # saving a remote LLM round-trip (the shared proxy can take 10-20s
        # per call under load). Ambiguous queries fall through to the ReAct
        # loop for LLM-driven tool selection.
        fast_decision = self._fast_route(query)
        if fast_decision is not None:
            self._add_step("⚡", "Routing",
                           "fast-route: %s (no routing LLM call)"
                           % fast_decision.get("tool"))
            return self._execute_decision(query, fast_decision, _stage,
                                          fallback=False)

        # Step 2: LangGraph ReAct loop (LLM native tool calling)
        _stage("routing", "classifying intent & selecting tool...")
        result = self._run_langgraph(query, history=history, _stage=_stage)
        if result is not None:
            return result

        # Step 3: deterministic fallback (router policy, no LLM involved)
        self._add_step("🧠", "Routing", "LLM tool calling unavailable -> router fallback")
        return self._run_from_router(query, _stage)

    # ── deterministic fast-route ─────────────────────────────────
    def _fast_route(self, query: str) -> Optional[Dict[str, Any]]:
        """Keyword-confident routing decision, or None when ambiguous.

        Confident cases (Candidate 2: risk now routes to the unified
        contract_search tool; the service applies the risk filters/ranking):
        - risk-keyword queries ("risk not accepted", "needs legal review")
          -> contract_search (risk intent preserved for the rank hint)
        - bare ref-number queries ("CCA20250096")
          -> contract_search (exact ref lookup happens inside the service)
        """
        intent = infer_intent_from_query(query)
        if intent == INTENT_RISK:
            tool = TOOL_CONTRACT_SEARCH
        elif _looks_like_ref_no(query):
            tool = TOOL_CONTRACT_SEARCH
            intent = INTENT_GENERAL
        else:
            return None
        decision = default_decision(query)
        decision["intent"] = intent
        decision["tool"] = tool
        return decision

    # ── LangGraph ReAct loop ────────────────────────────────────
    # 上限防护: 每个 tool 往返 ≈ 2-3 个 superstep; 10 步允许 ~3 轮工具调用,
    # 防止慢代理下失控循环 (超出即抛 GraphRecursionError → 走 router 兜底)。
    _RECURSION_LIMIT = 10

    def _run_langgraph(self, query: str, *,
                       history=None,
                       _stage=lambda l, d: None):
        """Run the LangGraph ReAct loop.  Returns None on failure."""
        if not (_LANGGRAPH_AVAILABLE and _BASE_CHAT_MODEL_AVAILABLE):
            logger.warning(
                "LangGraph unavailable (langgraph/langchain-core not importable); "
                "ReAct routing disabled, using deterministic router fallback")
            return None
        try:
            if self._react_agent is None:
                self._init_react_agent()
            if self._react_agent is None:
                return None
            messages = []
            sys_prompt = self._decision_system()
            if self.profile is not None:
                try:
                    sys_prompt = self.profile.build_system_prompt(sys_prompt)
                except Exception:
                    pass
            messages.append(("system", sys_prompt))
            if history:
                for turn in history:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if role == "user":
                        messages.append(("user", content))
                    elif role == "assistant":
                        messages.append(("assistant", content))
            # Hindsight on-demand recall (决策 16/17): 仅命中显式跨会话引用词
            # (上次/之前/还记得/remember/previously...) 才注入历史记忆; 未命中则静默跳过。
            user_content = query
            if self.hindsight_bank and needs_recall(query):
                _mem_t0 = time.perf_counter()
                memories: List[Dict[str, Any]] = []
                try:
                    memories = hs_recall(query, bank=self.hindsight_bank, max_results=5)
                except Exception as e:
                    logger.debug("Hindsight recall failed: %s", e)
                _mem_ms = int((time.perf_counter() - _mem_t0) * 1000)
                self._add_step(
                    "🧠", "Hindsight memory",
                    "recall on '%s' (%d memories, %dms)" % (query[:40], len(memories), _mem_ms),
                )
                if memories:
                    lines = []
                    for m in memories:
                        content = str(m.get("content") or "").strip()
                        if content:
                            lines.append("- " + content)
                    if lines:
                        user_content = (
                            query
                            + "\n\n[Relevant cross-session memories (Hindsight bank %s)]:\n%s\n"
                              "[Use these memories if relevant; ignore otherwise.]"
                            % (self.hindsight_bank, "\n".join(lines))
                        )
            messages.append(("user", user_content))
            _stage("running", "ReAct agent reasoning...")
            result = self._react_agent.invoke(
                {"messages": messages},
                config={"recursion_limit": self._RECURSION_LIMIT},
            )
            output_messages = result.get("messages", [])
            if not output_messages:
                return None

            # 工具调用归属: tool_call_id → ToolMessage content (观察结果)。
            observations: Dict[str, str] = {}
            for m in output_messages:
                if getattr(m, "type", "") == "tool":
                    observations[getattr(m, "tool_call_id", "")] = str(getattr(m, "content", "") or "")
            tool_calls: List[Dict[str, Any]] = []
            for m in output_messages:
                for t in (getattr(m, "tool_calls", None) or []):
                    args = t.get("args", {}) or {}
                    obs = observations.get(t.get("id", ""), "")
                    tool_calls.append({
                        "tool": t.get("name", ""),
                        "tool_input": args.get("query", args) if isinstance(args, dict) else args,
                        "filters": args.get("filters", {}) if isinstance(args, dict) else {},
                        "observation": obs[:200],
                    })

            final_content = ""
            for m in reversed(output_messages):
                if getattr(m, "type", "") == "ai" and m.content:
                    final_content = str(m.content).strip()
                    break
            if not final_content:
                return None

            last_observation = ""
            for m in reversed(output_messages):
                if getattr(m, "type", "") == "tool":
                    last_observation = str(getattr(m, "content", "") or "")
                    break

            if not tool_calls:
                # 模型未调用任何工具直接作答 → 视为澄清/直接回答。
                self._add_step("💬", "Clarify", final_content[:60])
                return self._result(
                    output=final_content, intent=INTENT_CLARIFY, tool=TOOL_NONE,
                    tool_calls=[], success=True, fallback=False,
                    clarify=True, observation="",
                )

            last_tool = tool_calls[-1]["tool"]
            self._add_step("🧠", "Routing", "LLM tool calling (ReAct): tool=%s" % last_tool)
            # Candidate 2: risk intent is inferred from the query, not the tool
            # (risk queries now route to the unified contract_search tool).
            intent = INTENT_RISK if infer_intent_from_query(query) == INTENT_RISK else INTENT_GENERAL
            return self._result(
                output=final_content,
                intent=intent,
                tool=last_tool,
                tool_calls=tool_calls,
                success=True,
                fallback=False,
                clarify=False,
                observation=last_observation,
            )
        except Exception as e:
            logger.warning("LangGraph agent failed: %s; falling back to router", e)
            return None

    def _init_react_agent(self):
        """Lazy-initialize the LangGraph ReAct agent."""
        if not (_LANGGRAPH_AVAILABLE and _BASE_CHAT_MODEL_AVAILABLE):
            self._react_agent = None
            return
        if self._react_agent is not None:
            return
        try:
            model = self._get_llm()
            if not isinstance(model, BaseChatModel):
                logger.warning("LLM %s is not a BaseChatModel; LangGraph disabled", type(model).__name__)
                self._react_agent = None
                return
            def _forward_tool_event(name, args):
                cb = self._on_tool_callback  # read current per-request callback
                if cb is not None:
                    cb(name, args)

            tools = [
                _notifying_tool(t, on_tool_callback=_forward_tool_event,
                                tool_meta_log=self._tool_meta_log)
                for t in self._tools
            ]
            # The legacy langgraph.prebuilt.create_react_agent path emits a
            # LangGraphDeprecatedSinceV10 warning at call time; suppress that
            # single known warning so it does not drown real test warnings.
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                self._react_agent = _create_react_agent(model, tools=tools)
        except Exception as e:
            logger.warning("Failed to init LangGraph agent: %s", e)
            self._react_agent = None

    # ── System prompt & deterministic filter inference ────────────
    # Shared status/contract-type label clause reused by the decision prompts
    # (kept as one fragment so a label change is made once, not per-prompt).
    _STATUS_CTYPE_CLAUSE = (
        "Status labels in the system: " + _STATUS_DESC + ". "
        "Contract types: " + _CTYPE_DESC + ". "
    )

    # Candidate 2: single unified prompt (the legacy two-tool _DECISION_SYSTEM
    # that told the LLM to "call risk_search" is removed; risk is handled by
    # the unified contract_search tool).
    _DECISION_SYSTEM_UNIFIED = (
        "You are a contract search assistant for a legal/OA system. "
        "Decide whether to call the contract_search tool for the user's message. "
        "Rules: "
        "1) Any genuine contract question -> call contract_search. This covers "
        "contract content (clauses, breach, termination, liability, amounts, "
        "counterparties, renewal/expiry, dates, status, expired, contract id) "
        "AND risk/compliance screening (risk not accepted, needs legal/GFN "
        "review, value thresholds, external guarantees, related-party/data/capex, "
        "unlimited liability, incomplete documentation, authority insufficient): "
        "risk filters and risk ranking are extracted automatically by the "
        "unified search service. "
        "2) When the query mentions status (e.g. completed, active, pending), "
        "expired contracts, or a specific contract id/ref number, include those "
        "values in the filters argument. "
        "3) " + _STATUS_CTYPE_CLAUSE +
        "4) Greetings, small talk, or vague/ambiguous messages -> do NOT call any "
        "tool; instead reply with a short clarifying question. "
        "Always prefer calling the tool for genuine contract questions."
    )

    _DECISION_SYSTEM_WHERE = _DECISION_SYSTEM_UNIFIED + (
        " "
        "5) Exact structured retrieval -> call contracts_where instead: "
        "'list all contracts with <exact condition>' requests (amount "
        "comparisons, date bounds, coded flag labels, status filters), and "
        "bare 'list all contracts' (empty condition returns every contract). "
        "Routing rule: 'list all'-style queries with no semantic content go "
        "to the structured path (contracts_where), never to vector search."
    )

    def _decision_system(self) -> str:
        """System prompt for tool selection, matching the wired tools.

        Candidate 2: there is no separate risk tool, so the unified prompt is
        always used; the where-tool variant adds the structured-retrieval rule.
        """
        if self._has_where_tool:
            return self._DECISION_SYSTEM_WHERE
        return self._DECISION_SYSTEM_UNIFIED

    # Filter vocabulary lives in the dedicated filters module (single source
    # of truth); these are thin back-compat references for existing callers.
    _DEPARTMENT_ALIASES = _FILTERS_DEPARTMENT_ALIASES
    _STATUS_ALIASES = _SHARED_STATUS_ALIASES
    _CONTRACT_TYPE_ALIASES = _FILTERS_CONTRACT_TYPE_ALIASES

    @staticmethod
    def _infer_contract_filters(query: str) -> Dict[str, Any]:
        """Deterministically extract contract facet filters from a query.

        Delegates to the filters module (single source of truth); kept as a
        method for back-compat with existing callers/tests.
        """
        return _infer_filters(query)

    # ── deterministic router fallback ──────────────────────────
    def _run_from_router(self, query: str, _stage) -> Dict[str, Any]:
        """Router fallback path: LLM router decision (deterministic default
        when the LLM fails) + execution, marked as the fallback path."""
        decision = self.router.decide(query)
        return self._execute_decision(query, decision, _stage, fallback=True)

    def _execute_decision(self, query: str, decision: Dict[str, Any], _stage,
                          fallback: bool) -> Dict[str, Any]:
        """Execute a routing decision: run the chosen tool, then synthesize
        the answer. Shared by the fast-route and the router fallback paths.

        fallback=True marks the result as the degraded path (the UI shows a
        warning banner); fast-route passes False since routing itself did not
        need the LLM."""
        intent = decision.get("intent", INTENT_GENERAL)
        tool = decision.get("tool") or TOOL_CONTRACT_SEARCH
        if tool == TOOL_CONTRACTS_WHERE and not self._has_where_tool:
            tool = TOOL_CONTRACT_SEARCH
        if tool not in (TOOL_CONTRACT_SEARCH, TOOL_CONTRACTS_WHERE):
            tool = TOOL_CONTRACT_SEARCH
        retrieval_query = decision.get("query") or query
        filters = decision.get("filters") or {}
        self._add_step("router", "Routing", "intent=%s, tool=%s" % (intent, tool))

        if not decision.get("search", True) or tool == TOOL_NONE:
            clarification = decision.get("clarification_question") or _DEFAULT_CLARIFICATION
            self._add_step("💬", "Clarify", clarification[:60])
            return self._result(
                output=clarification, intent=intent, tool=TOOL_NONE, tool_calls=[],
                success=True, fallback=fallback, clarify=True, observation="",
            )

        observation = ""
        tool_calls: List[Dict[str, Any]] = []
        try:
            # Candidate 2: risk intent routes to the unified contract_search
            # tool (the service extracts risk filters/ranking); contracts_where
            # is dispatched inside the ReAct loop, not on this fallback path.
            _stage("contract", "contract_search... query: %s" % retrieval_query)
            filters = dict(filters)
            inferred = self._infer_contract_filters(retrieval_query) or self._infer_contract_filters(query)
            for key, value in inferred.items():
                filters.setdefault(key, value)
            observation = self.contract_tool(retrieval_query, filters)
            self._add_step("search", "contract_search", "Query: '%s'" % retrieval_query)
            tool_calls.append({"tool": TOOL_CONTRACT_SEARCH, "tool_input": retrieval_query,
                               "filters": filters, "observation": observation[:200]})
        except Exception as e:
            logger.warning("LangChain agent tool failed: %s", e)
            self._add_step("warn", "Tool error", str(e)[:100])
            return self._result(
                output="I could not complete the search: %s" % e, intent=intent,
                tool=tool, tool_calls=tool_calls, success=False, fallback=True,
                clarify=False, observation=observation,
            )

        _stage("answer", "generating answer...")
        self._add_step("answer", "Answer generation", "Combining results")
        output = self._synthesize(query, tool, observation)
        return self._result(
            output=output, intent=intent, tool=tool, tool_calls=tool_calls,
            success=True, fallback=fallback, clarify=False, observation=observation,
        )
