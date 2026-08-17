"""Answer synthesis (Candidate 3).

Deep module that owns the one copy of the "turn a tool observation into a
natural-language answer" policy. It hides the messy bits behind a narrow
interface:

* the **prompt construction** (which wording, in which order);
* the **LLM call plumbing** (the model may be a raw ``fn(messages) -> str``
  callable or a LangChain chat model with ``.invoke(...)`` whose return value
  has a ``.content`` attribute);
* the **deterministic fallback** used when the LLM is unavailable (so the
  answer never echoes the raw prompt or the raw evidence).

The evidence text is sent to the model untruncated: it is already
budget-capped upstream by the observation formatter (a fixed row budget plus
an overflow marker), so capping it again here would silently drop contracts
from the summary. The full result set is rendered in the UI via the result
store, not via this prompt.

Both ``CrossTableAgent`` and ``LangChainAgent`` delegate here instead of
duplicating the logic, so a prompt change is made in exactly one place.

Run the tests:
    venv/bin/python -m pytest tests/test_synthesis.py -v
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

NL = chr(10)

# Deterministic messages (module constants so tests and callers can import the
# exact strings instead of re-typing them).
EMPTY_OBSERVATION_MESSAGE = "No matching contracts were found."
NO_LLM_MESSAGE = (
    "I found matching contracts, but the language model is unavailable, "
    "so I cannot summarize them right now. Browse the results below for details."
)

# The synthesis prompt is a single human message. Tests (and the LangChain
# message coercion) rely on this shape: ``messages[0] == ("human", prompt)``.
_PROMPT_TEMPLATE = (
    "Summarize the overall contract search results in 2-4 sentences." + NL +
    "Focus on the main patterns, common themes, and whether the results " + NL +
    "suggest approvals, renewals, risk flags, or other notable trends. " + NL +
    "Do not list record IDs, raw field names, or paste large excerpts. " + NL +
    "Use plain English and be concise." + NL + NL +
    "User query: {query}" + NL +
    "Tool: {tool}" + NL +
    "Evidence:" + NL + "{observation}"
)


def build_synthesis_prompt(query: str, tool: str, observation: str) -> str:
    """Return the human-message synthesis prompt (evidence sent untruncated)."""
    return _PROMPT_TEMPLATE.format(
        query=query, tool=tool, observation=observation or ""
    )


def _fallback_summary(observation: str) -> str:
    """Deterministic, human-readable summary when no LLM is available.

    Never returns the raw prompt or an evidence dump; reports the count of
    numbered evidence entries when it can.
    """
    obs = observation or ""
    n = obs.count("[")  # '[' begins each numbered evidence entry
    if n <= 0:
        return NO_LLM_MESSAGE
    return (
        f"I found {n} matching contract{'s' if n != 1 else ''}, but the language "
        "model is unavailable, so I cannot summarize them right now. Browse the "
        "results below for details."
    )


class AnswerSynthesizer:
    """Summarize a tool observation into a short natural-language answer.

    Parameters
    ----------
    llm:
        Optional LLM used for synthesis. Either a callable
        ``fn(messages) -> str`` (the legacy seam used by ``CrossTableAgent``)
        or a LangChain chat model exposing ``.invoke(messages)`` whose return
        value has a ``.content`` attribute. When ``None`` (or when it raises),
        the deterministic :func:`_fallback_summary` text is returned instead.
    """

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def synthesize(self, query: str, tool: str, observation: str) -> str:
        """Return a concise summary of ``observation`` for ``query``."""
        if not observation or not observation.strip():
            return EMPTY_OBSERVATION_MESSAGE

        llm = self._llm
        if llm is None:
            return _fallback_summary(observation)

        messages = [("human", build_synthesis_prompt(query, tool, observation))]
        try:
            # Callable seam (CrossTableAgent): fn(messages) -> str
            if callable(llm) and not hasattr(llm, "invoke"):
                text = llm(messages)
                text = (text or "").strip() if isinstance(text, str) else str(text).strip()
                if text:
                    return text
                return _fallback_summary(observation)
            # LangChain chat-model seam: .invoke(messages) -> obj with .content
            response = llm.invoke(messages)
            text = getattr(response, "content", "") or str(response)
            text = text.strip()
            if text:
                return text
        except Exception as e:
            logger.warning("LLM synthesis failed (%s); using fallback summary", e)
        return _fallback_summary(observation)

    @staticmethod
    def fallback_summary(observation: str) -> str:
        """Expose the deterministic fallback for callers/tests."""
        return _fallback_summary(observation)
