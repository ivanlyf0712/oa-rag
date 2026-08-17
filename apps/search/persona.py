"""
OA Search — Persona (CARA-style disposition layer)
====================================================
Tunable disposition traits (skepticism / literality / empathy / style) that
condition answer generation via system-prompt injection at the answer points.
Ported from corpchat-rag (apps/corpchat/search/persona.py).

Prompt conditioning only — no fine-tuning, no new runtime services. The neutral
default (0.5, balanced) appends no substantive instructions, so existing answers
are unchanged unless a profile is tuned.

The Hindsight bridge (`from_hindsight` / `sync_to_hindsight`) lets a Hindsight
memory bank's disposition (1-5 integers, edited in the Hindsight Web UI) drive
the agent's answer style: OA RAG reads the bank config and maps it to CARA 0-1
traits, degrading to the neutral default when Hindsight is unreachable.
"""

from dataclasses import dataclass
from typing import Dict, Optional

# Hindsight disposition 是 1–5 整数, CARA 是 0–1 浮点: v_0to1 = (v_1to5 - 1) / 4
_DISPOSITION_KEYS = {
    "skepticism": "disposition_skepticism",
    "literality": "disposition_literalism",
    "empathy": "disposition_empathy",
}


def _hindsight_disposition_to_profile(d: Dict) -> Dict:
    """Hindsight bank config → CARA 0-1 dict。缺失项用中性 0.5。

    兼容两种输入形状: 原始 bank config (disposition_skepticism/...) 与
    hindsight_client.get_disposition 的已重命名输出 (skepticism/...)。
    """
    out: Dict = {}
    for carakey, hs_key in _DISPOSITION_KEYS.items():
        v = d.get(hs_key, d.get(carakey))
        if v is None:
            out[carakey] = 0.5
        else:
            try:
                out[carakey] = max(0.0, min(1.0, (float(v) - 1.0) / 4.0))
            except (TypeError, ValueError):
                out[carakey] = 0.5
    out["style"] = "balanced"
    return out


@dataclass
class DispositionProfile:
    """CARA disposition profile: 0..1 traits + output style.

    - skepticism: 对检索证据不足的结论标注不确定性, 避免臆断
    - literality: 严格依据检索原文回答, 不添加推测
    - empathy:    先回应情绪/语气再给信息, 温和体谅
    - style:      concise / balanced / detailed
    """

    skepticism: float = 0.5
    literality: float = 0.5
    empathy: float = 0.5
    style: str = "balanced"

    # 低/中/高 阈值: 低于 LOW 或高于 HIGH 才追加对应指令
    LOW = 0.35
    HIGH = 0.65

    def _instructions(self) -> str:
        lines = []
        if self.skepticism >= self.HIGH:
            lines.append("- If the retrieved evidence is insufficient, explicitly "
                         "flag the uncertainty instead of drawing conclusions.")
        elif self.skepticism <= self.LOW:
            lines.append("- Draw conclusions directly from the retrieved evidence; "
                         "do not repeatedly hedge with uncertainty disclaimers.")
        if self.literality >= self.HIGH:
            lines.append("- Answer strictly from the retrieved source text; "
                         "do not add speculation or extrapolation.")
        elif self.literality <= self.LOW:
            lines.append("- You may summarise and extrapolate reasonably beyond "
                         "the source text; verbatim quoting is not required.")
        if self.empathy >= self.HIGH:
            lines.append("- Answer in a warm, considerate tone; acknowledge the "
                         "user's situation before giving information.")
        elif self.empathy <= self.LOW:
            lines.append("- Give information directly and concisely; "
                         "do not over-embellish the tone.")
        if self.style == "concise":
            lines.append("- Keep answers brief: conclusions and key points only.")
        elif self.style == "detailed":
            lines.append("- Answer in detail: background, key points, and "
                         "suggested follow-ups.")
        return "\n".join(lines)

    def build_system_prompt(self, base_prompt: str) -> str:
        """在基础 system prompt 后追加性格指令 (中性默认无实质追加)。"""
        if not base_prompt:
            return base_prompt
        instructions = self._instructions()
        if not instructions:
            return base_prompt
        return f"{base_prompt}\n\nAnswer style (per configured persona):\n{instructions}"

    def to_dict(self) -> Dict:
        return {
            "skepticism": self.skepticism,
            "literality": self.literality,
            "empathy": self.empathy,
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DispositionProfile":
        return cls(
            skepticism=float(d.get("skepticism", 0.5)),
            literality=float(d.get("literality", 0.5)),
            empathy=float(d.get("empathy", 0.5)),
            style=str(d.get("style", "balanced")),
        )

    @classmethod
    def from_hindsight(cls, bank_id: str,
                       api_url: Optional[str] = None) -> "DispositionProfile":
        """从 Hindsight bank 的 disposition 读取 CARA 人格。

        Hindsight 的每个 bank 有 disposition 三维度 (1–5):
          disposition_skepticism / disposition_literalism / disposition_empathy
        你在 Hindsight Web UI (Control Plane) 调整这些值, OA RAG 读取后
        映射为 CARA 0-1 人格。Hindsight 不可达时返回中性默认 (0.5)。

        Args:
            bank_id: Hindsight 记忆银行 ID (如 "oa-rag")
            api_url: 保留参数 (URL 统一由 hindsight_client 适配器解析)。
        """
        from apps.search import hindsight_client as hc
        d = hc.get_disposition(bank_id)
        if not d:
            # Hindsight 不可用 → 中性默认, 优雅降级
            return cls()
        return cls.from_dict(_hindsight_disposition_to_profile(d))

    def sync_to_hindsight(self, bank_id: str,
                          api_url: Optional[str] = None) -> bool:
        """把当前 CARA 人格写回 Hindsight bank 的 disposition (1–5)。

        让 OA RAG 里调好的人格同步到 Hindsight Web UI 可见。失败返回 False。
        """
        from apps.search import hindsight_client as hc
        return hc.set_disposition(
            skepticism=round(float(getattr(self, "skepticism", 0.5)) * 4.0 + 1.0),
            literality=round(float(getattr(self, "literality", 0.5)) * 4.0 + 1.0),
            empathy=round(float(getattr(self, "empathy", 0.5)) * 4.0 + 1.0),
            bank=bank_id,
        )
