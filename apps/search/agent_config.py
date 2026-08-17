"""
OA Search — Agent Configuration Model
======================================
Session-level agent configuration for the Settings page, ported from
corpchat-rag (apps/corpchat/search/agent_config.py) and trimmed to the knobs
the OA LangChain agent actually uses: the CARA persona + Hindsight bank
bridge. (corpchat's search-strategy / knowledge-scope panels have no OA
counterpart and are not ported.)

Single source of truth: `st.session_state.agent_config` (session-level,
immediate effect).

Values:
  - persona.skepticism / literality / empathy: 0-10 (UI slider scale)
  - persona.style: concise | balanced | detailed  (= answer length)
  - persona.hindsight_bank: optional Hindsight bank id; when set, the bank's
    disposition (edited in the Hindsight Web UI) drives the persona and the
    local sliders become a read-only mirror.
"""

from copy import deepcopy
from typing import Dict

# CARA 预设 (0-10 滑杆值 + 回答长度)
CARA_PRESETS: Dict[str, Dict] = {
    "audit":    {"skepticism": 8, "literality": 7, "empathy": 3, "style": "balanced"},
    "service":  {"skepticism": 3, "literality": 4, "empathy": 8, "style": "concise"},
    "research": {"skepticism": 6, "literality": 6, "empathy": 5, "style": "detailed"},
}

PRESET_LABELS = {
    "Audit Assistant": "audit",
    "Support Assistant": "service",
    "Research Assistant": "research",
    "Custom": "custom",
}

# 回答长度 (UI 标签 → style key), 顺序即下拉选项顺序
# (oa-rag persona.py 的中性风格词是 "balanced", 对应 corpchat 的 "standard")
STYLE_LABELS = {
    "Concise": "concise",
    "Standard": "balanced",
    "Detailed": "detailed",
}


def preset_index(preset_key: str) -> int:
    """预设 key → 下拉选项索引 (用于 st.selectbox index=)。"""
    for i, (label, key) in enumerate(PRESET_LABELS.items()):
        if key == preset_key:
            return i
    return len(PRESET_LABELS) - 1  # custom


def style_index(style: str) -> int:
    """style key → 下拉选项索引。"""
    for i, (label, key) in enumerate(STYLE_LABELS.items()):
        if key == style:
            return i
    return 1  # standard/balanced


_DEFAULT_CONFIG: Dict = {
    "persona": {
        "preset": "custom",
        "skepticism": 5,
        "literality": 5,
        "empathy": 5,
        "style": "balanced",
        "hindsight_bank": "",
    },
}


def default_agent_config() -> Dict:
    """返回默认配置的独立副本 (不共享可变状态)。"""
    return deepcopy(_DEFAULT_CONFIG)


def apply_preset(config: Dict, preset_label: str) -> Dict:
    """将 CARA 预设写入 config["persona"] (0-10 值); 'Custom' 不改值。"""
    key = PRESET_LABELS.get(preset_label, "custom")
    if key != "custom" and key in CARA_PRESETS:
        config["persona"].update(CARA_PRESETS[key])
    config["persona"]["preset"] = key
    return config


def persona_to_profile_dict(persona: Dict) -> Dict:
    """把 0-10 persona 值换算为 0-1 的 DispositionProfile 字典。"""
    return {
        "skepticism": float(persona.get("skepticism", 5)) / 10.0,
        "literality": float(persona.get("literality", 5)) / 10.0,
        "empathy": float(persona.get("empathy", 5)) / 10.0,
        "style": persona.get("style", "balanced"),
    }
