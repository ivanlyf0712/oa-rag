"""Regression gate for the Hindsight memory bridge (ported from corpchat-rag).

Covers, without any live Hindsight server / LLM / index / DB:
  - needs_recall gate (决策 16 trigger words; in-session pronouns excluded)
  - hindsight_client REST functions over a stubbed requests layer
  - DispositionProfile prompt injection + the Hindsight 1-5 <-> 0-1 bridge
  - LangChainAgent gated recall injection into the ReAct loop
  - app-level retain content building

Run:
    venv/bin/python -m pytest tests/test_hindsight.py -v
"""
import os
import sys
from typing import Any, Dict, List

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.search import hindsight_client as hc
from apps.search.persona import DispositionProfile


# ── 决策 16: 记忆触达词 gate ──────────────────────────────────────
class TestNeedsRecallGate:
    """命中显式跨会话引用词才需要 Hindsight recall; 其余静默跳过。"""

    @pytest.mark.parametrize("q", [
        "上次我们查过的合同",
        "上回说的那个客户",
        "之前聊过的续约条款",
        "以前签的那份协议",
        "还记得那个报价吗",
        "记得当时的结论",
        "那次查到的风险",
        "那件事后来怎么样了",
    ])
    def test_chinese_triggers(self, q):
        assert hc.needs_recall(q), f"{q!r} should trigger recall"

    @pytest.mark.parametrize("q", [
        "do you remember the contract we signed",
        "what did we find last time",
        "as i said before, check the clause",
        "we discussed this previously",
        "show the earlier results",
    ])
    def test_english_triggers(self, q):
        assert hc.needs_recall(q), f"{q!r} should trigger recall"

    @pytest.mark.parametrize("q", [
        "show completed contracts with Acme",
        "which contracts mention unlimited liability",
        "is CCA20250096 expired",
        "high value agreements signed this year",
    ])
    def test_plain_queries_skip(self, q):
        assert not hc.needs_recall(q), f"{q!r} must NOT trigger recall"

    @pytest.mark.parametrize("q", [
        "她的邮箱是什么",
        "这个合同的金额是多少",
        "那个供应商的联系方式",
    ])
    def test_in_session_pronouns_skip(self, q):
        """会话内指代 (她/这个/那个) 走历史注入, 不触发 Hindsight recall。"""
        assert not hc.needs_recall(q), f"{q!r} must NOT trigger recall"

    def test_english_word_boundary(self):
        """整词匹配: "before" 不得命中 "beforehand"。"""
        assert not hc.needs_recall("search beforehand please")

    def test_empty_query(self):
        assert not hc.needs_recall("")
        assert not hc.needs_recall(None)


# ── REST client (stubbed requests) ───────────────────────────────
class _FakeResp:
    def __init__(self, status: int = 200, payload: Any = None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _FakeRequests:
    """Records calls and serves scripted responses keyed by (method, path)."""

    def __init__(self, responses: Dict = None, raise_exc: bool = False):
        self.calls: List[Dict[str, Any]] = []
        self.responses = responses or {}
        self.raise_exc = raise_exc

    def _handle(self, method: str, url: str, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        if self.raise_exc:
            raise RuntimeError("connection refused")
        for (m, needle), resp in self.responses.items():
            if m == method and needle in url:
                return resp
        return _FakeResp(404)

    def post(self, url, **kw):
        return self._handle("POST", url, **kw)

    def get(self, url, **kw):
        return self._handle("GET", url, **kw)

    def patch(self, url, **kw):
        return self._handle("PATCH", url, **kw)


class TestRetain:
    def test_success_posts_items(self, monkeypatch):
        fake = _FakeRequests({("POST", "/memories"): _FakeResp(200)})
        monkeypatch.setattr(hc, "requests", fake)
        ok = hc.retain("turn summary", bank="b1", context="oa-rag-search",
                       document_id="doc-1", tags=["oa-rag"], async_=True)
        assert ok is True
        call = fake.calls[0]
        assert "/v1/default/banks/b1/memories" in call["url"]
        assert call["json"]["async"] is True
        item = call["json"]["items"][0]
        assert item["content"] == "turn summary"
        assert item["context"] == "oa-rag-search"
        assert item["document_id"] == "doc-1"
        assert item["tags"] == ["oa-rag"]

    def test_failure_status_returns_false(self, monkeypatch):
        fake = _FakeRequests({("POST", "/memories"): _FakeResp(500)})
        monkeypatch.setattr(hc, "requests", fake)
        assert hc.retain("x", bank="b1") is False

    def test_exception_returns_false(self, monkeypatch):
        monkeypatch.setattr(hc, "requests", _FakeRequests(raise_exc=True))
        assert hc.retain("x", bank="b1") is False


class TestRecall:
    def test_parses_results_and_unifies_content(self, monkeypatch):
        payload = {"results": [
            {"id": "1", "text": "Acme 偏好电话沟通"},
            {"id": "2", "content": "CCA20250096 已过期"},
        ]}
        fake = _FakeRequests({("POST", "/memories/recall"): _FakeResp(200, payload)})
        monkeypatch.setattr(hc, "requests", fake)
        out = hc.recall("q", bank="b1", max_results=5)
        assert [m["content"] for m in out] == ["Acme 偏好电话沟通", "CCA20250096 已过期"]
        call = fake.calls[0]
        assert "/v1/default/banks/b1/memories/recall" in call["url"]
        assert call["json"] == {"query": "q"}
        assert call["params"] == {"limit": 5}

    def test_respects_max_results(self, monkeypatch):
        payload = {"results": [{"text": "m%d" % i} for i in range(10)]}
        monkeypatch.setattr(hc, "requests",
                            _FakeRequests({("POST", "/recall"): _FakeResp(200, payload)}))
        assert len(hc.recall("q", bank="b1", max_results=3)) == 3

    def test_non_200_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hc, "requests", _FakeRequests())
        assert hc.recall("q", bank="b1") == []

    def test_exception_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hc, "requests", _FakeRequests(raise_exc=True))
        assert hc.recall("q", bank="b1") == []


class TestReflect:
    def test_returns_answer(self, monkeypatch):
        fake = _FakeRequests({("POST", "/reflect"): _FakeResp(200, {"answer": " grounded "})})
        monkeypatch.setattr(hc, "requests", fake)
        assert hc.reflect("q", bank="b1") == " grounded "

    def test_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hc, "requests", _FakeRequests(raise_exc=True))
        assert hc.reflect("q", bank="b1") == ""


class TestEntityGraphAndBanks:
    def test_graph_flattens_data_wrappers(self, monkeypatch):
        payload = {
            "nodes": [{"data": {"id": "Acme"}}, {"data": {"id": "CCA1"}}],
            "edges": [{"data": {"source": "Acme", "target": "CCA1"}}],
            "total_entities": 2, "total_edges": 1,
        }
        fake = _FakeRequests({("GET", "/entities/graph"): _FakeResp(200, payload)})
        monkeypatch.setattr(hc, "requests", fake)
        g = hc.get_entity_graph("b1", limit=10)
        assert g["total_entities"] == 2
        assert g["nodes"] == [{"id": "Acme"}, {"id": "CCA1"}]
        assert g["edges"] == [{"source": "Acme", "target": "CCA1"}]

    def test_graph_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hc, "requests", _FakeRequests(raise_exc=True))
        assert hc.get_entity_graph("b1") == {}

    def test_list_banks(self, monkeypatch):
        payload = {"banks": [{"bank_id": "a"}, {"bank_id": "b"}, {"nope": 1}]}
        fake = _FakeRequests({("GET", "/v1/default/banks"): _FakeResp(200, payload)})
        monkeypatch.setattr(hc, "requests", fake)
        assert hc.list_banks() == ["a", "b"]


class TestDispositionBridge:
    def test_get_disposition_renames_keys(self, monkeypatch):
        payload = {"config": {
            "disposition_skepticism": 5,
            "disposition_literalism": 1,
            "disposition_empathy": 3,
        }}
        fake = _FakeRequests({("GET", "/config"): _FakeResp(200, payload)})
        monkeypatch.setattr(hc, "requests", fake)
        d = hc.get_disposition("b1")
        assert d == {"skepticism": 5, "literality": 1, "empathy": 3}

    def test_get_disposition_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hc, "requests", _FakeRequests(raise_exc=True))
        assert hc.get_disposition("b1") == {}

    def test_set_disposition_clamps_to_1_5(self, monkeypatch):
        fake = _FakeRequests({("PATCH", "/config"): _FakeResp(200)})
        monkeypatch.setattr(hc, "requests", fake)
        ok = hc.set_disposition(skepticism=0, literality=9, empathy=3, bank="b1")
        assert ok is True
        updates = fake.calls[0]["json"]["updates"]
        assert updates == {
            "disposition_skepticism": 1,
            "disposition_literalism": 5,
            "disposition_empathy": 3,
        }


def test_env_overrides_url_and_bank(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://hs.internal:9999/")
    monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")
    assert hc._url("/config") == "http://hs.internal:9999/v1/default/banks/env-bank/config"
    assert hc._url("/config", bank="explicit") == \
        "http://hs.internal:9999/v1/default/banks/explicit/config"


# ── Persona (DispositionProfile) ─────────────────────────────────
BASE_PROMPT = "You are an OA contract search assistant."


def test_default_profile_is_neutral():
    p = DispositionProfile()
    assert p.build_system_prompt(BASE_PROMPT) == BASE_PROMPT


def test_high_skepticism_adds_uncertainty_instruction():
    p = DispositionProfile(skepticism=0.9)
    out = p.build_system_prompt(BASE_PROMPT)
    assert out.startswith(BASE_PROMPT)
    assert "uncertainty" in out


def test_low_skepticism_adds_direct_instruction():
    p = DispositionProfile(skepticism=0.1)
    assert "do not repeatedly hedge" in p.build_system_prompt(BASE_PROMPT)


def test_high_literality_adds_literal_instruction():
    p = DispositionProfile(literality=0.9)
    assert "strictly from the retrieved source text" in p.build_system_prompt(BASE_PROMPT)


def test_high_empathy_adds_empathetic_instruction():
    p = DispositionProfile(empathy=0.9)
    assert "warm, considerate tone" in p.build_system_prompt(BASE_PROMPT)


def test_concise_style_adds_brevity_instruction():
    p = DispositionProfile(style="concise")
    assert "Keep answers brief" in p.build_system_prompt(BASE_PROMPT)


def test_detailed_style_adds_detail_instruction():
    p = DispositionProfile(style="detailed")
    assert "Answer in detail" in p.build_system_prompt(BASE_PROMPT)


def test_empty_base_prompt_passthrough():
    assert DispositionProfile(skepticism=0.9).build_system_prompt("") == ""


def test_profile_serializes_to_dict_and_back():
    p = DispositionProfile(skepticism=0.8, literality=0.2, empathy=0.6, style="detailed")
    p2 = DispositionProfile.from_dict(p.to_dict())
    assert p2 == p


def test_from_hindsight_consumes_adapter(monkeypatch):
    """from_hindsight 经 hindsight_client.get_disposition 读取, 映射 1-5 → 0-1。"""
    captured = {}

    def _fake_get_disposition(bank=None):
        captured["bank"] = bank
        return {"skepticism": 5, "literality": 1, "empathy": 3}

    monkeypatch.setattr(hc, "get_disposition", _fake_get_disposition)

    profile = DispositionProfile.from_hindsight("oa-bank")

    assert captured["bank"] == "oa-bank"
    assert profile.skepticism == 1.0   # (5-1)/4
    assert profile.literality == 0.0   # (1-1)/4
    assert profile.empathy == 0.5      # (3-1)/4
    assert profile.style == "balanced"


def test_from_hindsight_unreachable_returns_neutral(monkeypatch):
    monkeypatch.setattr(hc, "get_disposition", lambda bank=None: {})
    assert DispositionProfile.from_hindsight("oa-bank") == DispositionProfile()


def test_from_hindsight_accepts_raw_bank_config_shape(monkeypatch):
    """原始 bank config (disposition_skepticism/...) 形状同样可用。"""
    monkeypatch.setattr(hc, "get_disposition", lambda bank=None: {
        "disposition_skepticism": 1, "disposition_literalism": 5,
        "disposition_empathy": None,
    })
    profile = DispositionProfile.from_hindsight("oa-bank")
    assert profile.skepticism == 0.0
    assert profile.literality == 1.0
    assert profile.empathy == 0.5  # 缺失 → 中性


def test_sync_to_hindsight_consumes_adapter(monkeypatch):
    """sync_to_hindsight 经 hindsight_client.set_disposition 写回 (0-1 → 1-5)。"""
    captured = {}

    def _fake_set_disposition(skepticism=None, literality=None, empathy=None, bank=None):
        captured.update(skepticism=skepticism, literality=literality,
                        empathy=empathy, bank=bank)
        return True

    monkeypatch.setattr(hc, "set_disposition", _fake_set_disposition)

    ok = DispositionProfile(skepticism=1.0, literality=0.0, empathy=0.5).sync_to_hindsight("b1")

    assert ok is True
    assert captured["bank"] == "b1"
    assert captured["skepticism"] == 5   # 1.0*4+1
    assert captured["literality"] == 1   # 0.0*4+1
    assert captured["empathy"] == 3      # 0.5*4+1


# ── LangChainAgent: gated recall injection (决策 16/17) ──────────
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseChatModel = object  # type: ignore[assignment]


class _EchoLLM(BaseChatModel):
    """Minimal scripted model: forwards the last human message into a
    contract_search call; once a ToolMessage exists, returns the answer.

    This lets the test observe exactly what user content the ReAct loop
    received (the tool arg == the final human message, memory block included).
    """

    bound_tools: List[Any] = []
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    @staticmethod
    def _result(message):
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return self._result(AIMessage(content="Scripted answer based on tool results."))
        query = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                query = str(m.content)
                break
        return self._result(AIMessage(content="", tool_calls=[{
            "name": "contract_search",
            "args": {"query": query, "filters": {}},
            "id": "call_%d" % self.calls}]))

    @property
    def _llm_type(self) -> str:
        return "echo-fake"


class _RecordingTools:
    def __init__(self):
        self.contract_calls: List[Dict[str, Any]] = []
        self.risk_calls: List[Dict[str, Any]] = []

    def contract(self, query, filters=None):
        self.contract_calls.append({"query": query, "filters": filters})
        return "CONTRACT RESULTS"

    def risk(self, query, filters=None):
        self.risk_calls.append({"query": query, "filters": filters})
        return "RISK RESULTS"


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
class TestAgentRecallGating:
    """决策 16/17: process() 只对命中触达词的查询做 Hindsight recall,
    命中时把记忆注入 ReAct 用户消息并记录时间线步骤; 未命中静默跳过。"""

    def _make_agent(self, tools, bank="test-bank"):
        from apps.search.langchain_agent import LangChainAgent
        return LangChainAgent(
            contract_tool=tools.contract, risk_tool=tools.risk,
            llm=_EchoLLM(), hindsight_bank=bank,
            synthesize=lambda q, t, o: "ANSWER: %s" % o,
        )

    def _patch_recall(self, monkeypatch, memories=None, raises=False):
        calls: List[Dict[str, Any]] = []

        def fake_recall(query, bank=None, max_results=5):
            calls.append({"query": query, "bank": bank, "max_results": max_results})
            if raises:
                raise RuntimeError("hindsight down")
            return list(memories or [])

        monkeypatch.setattr("apps.search.langchain_agent.hs_recall", fake_recall)
        return calls

    def test_trigger_query_calls_recall_and_logs_step(self, monkeypatch):
        calls = self._patch_recall(monkeypatch)
        agent = self._make_agent(_RecordingTools())
        out = agent.process("还记得我们之前聊过的续约条款吗")
        assert calls, "trigger-word query must call Hindsight recall"
        assert calls[0]["bank"] == "test-bank"
        assert any(s.get("label") == "Hindsight memory" for s in out["steps"])
        assert out["success"] is True

    def test_plain_query_skips_recall_silently(self, monkeypatch):
        calls = self._patch_recall(monkeypatch)
        agent = self._make_agent(_RecordingTools())
        out = agent.process("show completed contracts with Acme")
        assert not calls, "plain query must skip Hindsight recall"
        assert not any(s.get("label") == "Hindsight memory" for s in out["steps"])
        assert out["success"] is True

    def test_in_session_pronoun_skips_recall(self, monkeypatch):
        calls = self._patch_recall(monkeypatch)
        agent = self._make_agent(_RecordingTools())
        agent.process("她的联系方式是什么")
        assert not calls, "in-session pronoun must skip Hindsight recall"

    def test_no_bank_skips_recall(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_BANK_ID", raising=False)
        calls = self._patch_recall(monkeypatch)
        agent = self._make_agent(_RecordingTools(), bank=None)
        out = agent.process("还记得我们之前聊过的续约条款吗")
        assert agent.hindsight_bank is None
        assert not calls, "no bank configured → recall disabled"
        assert not any(s.get("label") == "Hindsight memory" for s in out["steps"])

    def test_bank_falls_back_to_env(self, monkeypatch):
        from apps.search.langchain_agent import LangChainAgent
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")
        agent = LangChainAgent(contract_tool=_RecordingTools().contract, llm=_EchoLLM())
        assert agent.hindsight_bank == "env-bank"

    def test_recall_failure_degrades_gracefully(self, monkeypatch):
        calls = self._patch_recall(monkeypatch, raises=True)
        agent = self._make_agent(_RecordingTools())
        out = agent.process("还记得我们之前聊过的续约条款吗")
        assert calls, "recall was attempted"
        assert out["success"] is True, "recall failure must not break the turn"

    def test_recalled_memory_injected_into_react_input(self, monkeypatch):
        self._patch_recall(monkeypatch, memories=[
            {"content": "Acme 偏好电话沟通"},
            {"content": "  "},          # blank entries are dropped
            {"no_content_key": True},     # entries without content are dropped
        ])
        tools = _RecordingTools()
        agent = self._make_agent(tools)
        agent.process("还记得我们之前聊过的续约条款吗")
        assert tools.contract_calls, "contract tool must have run"
        forwarded = tools.contract_calls[0]["query"]
        assert "还记得我们之前聊过的续约条款吗" in forwarded
        assert "Acme 偏好电话沟通" in forwarded
        assert "Relevant cross-session memories" in forwarded
        assert "test-bank" in forwarded


# ── app-level retain (apps.app helpers) ──────────────────────────
_ROWS = [
    {"metadata": {"ref_no": "CCA20250001", "counterparty_name": "Acme Ltd",
                  "title": "Master Services Agreement"}},
    {"metadata": {"ref_no": "CCA20250002", "counterparty_name": "Beta Corp",
                  "title": "NDA"}},
    {"metadata": {"ref_no": "CCA20250003", "counterparty_name": "Acme Ltd",
                  "title": "Renewal"}},
    {"metadata": {"ref_no": "CCA20250004", "counterparty_name": "Delta Inc",
                  "title": "PO"}},
]


def test_build_retain_content_includes_query_and_top_rows():
    from apps.app import _build_retain_content
    content = _build_retain_content("show Acme contracts", _ROWS)
    assert "show Acme contracts" in content
    assert "CCA20250001" in content and "Acme Ltd" in content
    # capped at max_rows=3
    assert "CCA20250003" in content
    assert "CCA20250004" not in content


def test_build_retain_content_handles_flat_rows_and_empty():
    from apps.app import _build_retain_content
    content = _build_retain_content("q", [{"ref_no": "CCA9"}])
    assert "CCA9" in content
    assert "q" in _build_retain_content("q", [])
    assert "User query:" in _build_retain_content("", [])


def test_retain_turn_tags_entity_anchors(monkeypatch):
    import apps.app as app_mod
    captured: Dict[str, Any] = {}

    def fake_retain(content, bank=None, **kw):
        captured.update(content=content, bank=bank, **kw)
        return True

    monkeypatch.setattr(app_mod, "hs_retain", fake_retain)
    ok = app_mod._memory()._retain("show Acme contracts", _ROWS, "b1")
    assert ok is True
    assert captured["bank"] == "b1"
    assert captured["async_"] is True
    assert captured["context"] == "oa-rag-search"
    tags = captured["tags"]
    assert tags[:2] == ["oa-rag", "search"]
    assert "Acme Ltd" in tags and "CCA20250001" in tags
    # deduped: Acme Ltd appears twice in rows, once in tags
    assert tags.count("Acme Ltd") == 1


def test_retain_turn_propagates_failure(monkeypatch):
    import apps.app as app_mod
    monkeypatch.setattr(app_mod, "hs_retain", lambda *a, **kw: False)
    assert app_mod._memory()._retain("q", [], "b1") is False


# ── agent_config (Settings page persona machinery, ported from corpchat) ──
class TestAgentConfig:
    def test_default_config_is_neutral_and_custom(self):
        from apps.search.agent_config import default_agent_config
        p = default_agent_config()["persona"]
        assert (p["skepticism"], p["literality"], p["empathy"]) == (5, 5, 5)
        assert p["style"] == "balanced" and p["preset"] == "custom"
        assert p["hindsight_bank"] == ""

    def test_default_config_returns_independent_copies(self):
        from apps.search.agent_config import default_agent_config
        a, b = default_agent_config(), default_agent_config()
        a["persona"]["skepticism"] = 9
        assert b["persona"]["skepticism"] == 5
        assert default_agent_config()["persona"]["skepticism"] == 5

    def test_apply_preset_writes_cara_values(self):
        from apps.search.agent_config import apply_preset, default_agent_config
        cfg = apply_preset(default_agent_config(), "Audit Assistant")
        assert cfg["persona"]["skepticism"] == 8
        assert cfg["persona"]["literality"] == 7
        assert cfg["persona"]["empathy"] == 3
        assert cfg["persona"]["style"] == "balanced"
        assert cfg["persona"]["preset"] == "audit"

    def test_apply_preset_custom_keeps_values(self):
        from apps.search.agent_config import apply_preset, default_agent_config
        cfg = default_agent_config()
        cfg["persona"]["skepticism"] = 2
        cfg = apply_preset(cfg, "Custom")
        assert cfg["persona"]["skepticism"] == 2
        assert cfg["persona"]["preset"] == "custom"

    def test_persona_to_profile_dict_scales_0_10_to_0_1(self):
        from apps.search.agent_config import persona_to_profile_dict
        d = persona_to_profile_dict({"skepticism": 10, "literality": 0,
                                     "empathy": 5, "style": "concise"})
        assert d == {"skepticism": 1.0, "literality": 0.0,
                     "empathy": 0.5, "style": "concise"}

    def test_local_sliders_feed_disposition_profile(self):
        from apps.search.agent_config import default_agent_config, persona_to_profile_dict
        from apps.search.persona import DispositionProfile
        prof = DispositionProfile.from_dict(
            persona_to_profile_dict(default_agent_config()["persona"]))
        assert prof == DispositionProfile(), "neutral sliders → neutral profile"
        assert prof.build_system_prompt("BASE") == "BASE", "neutral adds nothing"

    def test_preset_and_style_index_fallbacks(self):
        from apps.search.agent_config import (PRESET_LABELS, STYLE_LABELS,
                                              preset_index, style_index)
        assert preset_index("audit") == 0
        assert preset_index("nope") == len(PRESET_LABELS) - 1
        assert style_index("concise") == 0
        assert style_index("nope") == 1
        assert STYLE_LABELS["Standard"] == "balanced"


# ── app-level recall/skip indicator (mirrors corpchat Process window) ──
class TestRecallParticipationIndicator:
    """After an Ask turn, result['hindsight']['recall'] must be 'recall' when
    the agent fired a Hindsight memory recall step, 'skip' when the bank is
    configured but the gate did not fire."""

    def _drive(self, monkeypatch, steps, bank="b1"):
        import types
        import apps.app as app_mod

        captions: List[str] = []
        class _CM:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __getattr__(self, name):
                return lambda *a, **k: self

        st = types.SimpleNamespace()
        st.session_state = {"hindsight_bank": bank}
        st.subheader = lambda *a, **k: None
        st.caption = lambda text, *a, **k: captions.append(str(text))
        st.columns = lambda n, *a, **k: [_CM()
            for _ in range(n if isinstance(n, int) else len(n))]
        st.text_input = lambda *a, **k: k.get("value", "")
        st.button = lambda *a, **k: True          # Ask pressed
        st.empty = lambda *a, **k: types.SimpleNamespace(
            markdown=lambda *a, **k: None, empty=lambda: None)
        st.spinner = lambda *a, **k: _CM()
        st.expander = lambda *a, **k: _CM()
        st.markdown = lambda *a, **k: None
        st.warning = lambda *a, **k: None
        st.dataframe = lambda *a, **k: None
        st.error = lambda *a, **k: None
        monkeypatch.setattr(app_mod, "st", st)

        class _Agent:
            def process(self, query, on_stage=None):
                return {"output": "ok", "intent": "g", "tool": "contract_search",
                        "tool_calls": [], "steps": steps, "success": True,
                        "fallback": False}

        class _Searcher:
            def __init__(self, emb):
                pass

        monkeypatch.setattr(app_mod, "_load_embeddings", lambda p: object())
        monkeypatch.setattr(app_mod, "Searcher", _Searcher)
        monkeypatch.setattr(app_mod, "_build_agent", lambda p, e: _Agent())
        # retain through the Memory module's seam (patched at the app boundary)
        monkeypatch.setattr(app_mod, "hs_retain", lambda *a, **kw: True)
        # keep the persona profile load hermetic (no live Hindsight server)
        from apps.search.memory import Memory as _Mem
        monkeypatch.setattr(_Mem, "_load_profile",
                            lambda self, b: DispositionProfile())
        monkeypatch.setattr(app_mod, "snapshot_results", lambda: {"rows": []})
        monkeypatch.setattr(app_mod, "clear_results", lambda: None)
        # result rendering path expects the result store + contract renderer
        monkeypatch.setattr(app_mod, "_render_agentic_contract", lambda r, s: None)
        return app_mod, captions

    def test_recall_step_marks_recall(self, monkeypatch):
        steps = [{"icon": "\U0001F9E0", "label": "Hindsight memory",
                  "detail": "recall on 'x' (2 memories, 10ms)"}]
        app_mod, captions = self._drive(monkeypatch, steps)
        app_mod._render_agentic("/tmp/x")
        hs = app_mod.st.session_state["agentic_result"]["hindsight"]
        assert hs["recall"] == "recall"
        assert hs["retained"] is True
        assert any("memory recall used" in c for c in captions)

    def test_no_recall_step_marks_skip(self, monkeypatch):
        steps = [{"icon": "\U0001F9ED", "label": "Routing", "detail": "intent=g"}]
        app_mod, captions = self._drive(monkeypatch, steps)
        app_mod._render_agentic("/tmp/x")
        hs = app_mod.st.session_state["agentic_result"]["hindsight"]
        assert hs["recall"] == "skip"
        assert any("recall skipped" in c for c in captions)

    def test_no_bank_marks_no_hindsight_block(self, monkeypatch):
        app_mod, captions = self._drive(monkeypatch, [], bank="")
        app_mod._render_agentic("/tmp/x")
        result = app_mod.st.session_state["agentic_result"]
        assert "hindsight" not in result
        assert not any("Memory:" in c for c in captions)
