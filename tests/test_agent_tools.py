"""Tests for the agent tool wiring in apps/search_cli.py (Ticket 2).

Covers the real contract_search tool builder and the result-formatting
helpers, using in-memory fakes (no live index/DB/LLM).

Run:
    venv/bin/python -m pytest tests/test_agent_tools.py -v
"""
import os
import sys

import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import apps.search_cli as cli


# -- formatting helpers ------------------------------------------------
def test_format_contract_results_empty():
    assert cli.format_contract_results([]) == ""


def test_format_contract_results_rows():
    results = [
        {"text": "body text here", "metadata": {"counterparty_name": "Acme", "contract_type": 2}},
        {"text": "more text", "metadata": {"title": "T", "contract_type": 0}},
    ]
    out = cli.format_contract_results(results)
    assert "Acme" in out
    assert "body text here" in out
    assert out.count(chr(10)) == 1  # two lines


class FakeSearcher:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self._results


def test_contract_tool_passes_filters_to_searcher():
    results = [{"text": "x", "metadata": {"counterparty_name": "Acme Corp", "contract_type": 2}}]
    searcher = FakeSearcher(results)
    tool = cli.build_contract_tool(embeddings=None, searcher=searcher)
    out = tool("liability", {"contract_type": "2", "counterparty_name": "Acme"})
    kwargs = searcher.calls[0]
    assert kwargs["label_filter"] == "2"
    assert "Acme" in out


def test_contract_tool_empty_when_no_results():
    searcher = FakeSearcher([])
    tool = cli.build_contract_tool(embeddings=None, searcher=searcher)
    assert tool("nothing", {}) == ""
