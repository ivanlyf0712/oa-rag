#!/usr/bin/env python3
"""
CorpChat Search CLI — Onyx-style contract search command-line entrypoint.

This module is only the CLI (click). The retrieval stack and unified contract
search service live in the apps/search package; this file parses args and
calls the package API.

Usage:
  python apps/search_cli.py build [--force] [--graph-mode auto|llm|off]
  python apps/search_cli.py search "合同 违约责任" --mode hybrid --expand
  python apps/search_cli.py benchmark --runs 20
  python apps/search_cli.py agent "show me completed contracts with Alpha Corp"
"""

import logging
import os
import sys

# Allow running as a script (python apps/search_cli.py): put repo root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import statistics
import time
from typing import Dict, List

import pandas as pd
import click
from tabulate import tabulate

from apps.search import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_INDEX_PATH,
    DEFAULT_RERANK_TOP_N,
    LITELLM_MODEL,
    AgenticDecider,
    ContractSearchService,
    OBSERVATION_ROW_BUDGET,
    UnifiedQueryPlanner,
    format_contract_observation,
    format_contract_results,
    IndexBuilder,
    QueryExpander,
    Reranker,
    Searcher,
    load_index,
)
from apps.search._core import _clean_text_from_enriched
from apps.risk_search import RiskPlanner, run_risk_search
from apps.search.result_store import stash_results

logger = logging.getLogger("oa-search")


SYNTHETIC_TEST_QUERIES = [
    {"query": "purchase contract", "expected_contract_types": [0, 2, 3, 4],
     "expected_counterparties": [], "description": "general purchase / sales contracts"},
    {"query": "renewal contract", "expected_contract_types": [2],
     "expected_counterparties": [], "description": "renewal-style contracts"},
    {"query": "IT 系统 升级", "expected_contract_types": [2],
     "expected_counterparties": ["Shenzhen"], "description": "IT system upgrade service contracts"},
    {"query": "termination breach", "expected_contract_types": [0, 2, 3, 4],
     "expected_counterparties": [], "description": "breach / termination clauses"},
]


# ═══════════════════════════════════════════════════════════════════
# 9. CLI (click)
# ═══════════════════════════════════════════════════════════════════


def _format_results(results: List[Dict], show_len: int = 100) -> str:
    if not results:
        return "没有找到结果。\n"
    rows = []
    for i, r in enumerate(results, 1):
        text = r.get("text", "")
        meta = r.get("metadata", {})
        text_preview = _clean_text_from_enriched(text)[:show_len] + "..." if len(text) > show_len else _clean_text_from_enriched(text)
        graph_info = ""
        if meta.get("_graph_relation"):
            graph_info = f"🕸️ {meta['_graph_relation']}"
        if r.get("rerank_score") is not None:
            graph_info += f" [Rerank: {r['rerank_score']:.4f}]"

        rows.append([
            i,
            r["id"][:25],
            f"{r.get('score', 0):.4f}",
            str(meta.get("counterparty_name", "") or meta.get("title", ""))[:12],
            str(meta.get("contract_type", "-")),
            text_preview,
            graph_info,
        ])

    return tabulate(
        rows,
        headers=["#", "ID", "Score", "Counterparty", "Contract Type", "Content", "Info"],
        tablefmt="simple_grid",
        maxcolwidths=[None, 18, None, 10, 12, 55, 25],
    )


TEST_QUERIES = [
    {"query": "purchase contract", "expected_ids": [], "description": "purchase and sales contracts"},
    {"query": "renewal contract", "expected_ids": [], "description": "renewal contracts"},
    {"query": "IT 系统 升级", "expected_ids": [], "description": "IT upgrade services"},
    {"query": "termination breach", "expected_ids": [], "description": "termination and breach clauses"},
]


def _calc_mrr(predictions: List[str], expected: List[str]) -> float:
    for i, pid in enumerate(predictions, 1):
        if pid in expected:
            return 1.0 / i
    return 0.0


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command("build")
@click.option("--force", is_flag=True)
@click.option("--graph-mode", type=click.Choice(["auto", "llm", "off"]), default="auto")
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
@click.option("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int)
def build_cmd(force, graph_mode, index_path, chunk_size):
    try:
        enable_graph = graph_mode != "off"
        builder = IndexBuilder(index_path, chunk_size=chunk_size)
        embeddings = builder.build(force=force, enable_graph=enable_graph, graph_mode=graph_mode)
        click.echo(f"✅ 索引就绪 — {embeddings.count()} 个块 | 图: {'✅' if embeddings.graph else '❌'}")
    except Exception as e:
        logger.exception("构建失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("search")
@click.argument("query")
@click.option("--mode", default="hybrid", type=click.Choice(["keyword", "semantic", "hybrid"]))
@click.option("--limit", default=10, type=int)
@click.option("--expand/--no-expand", default=False)
@click.option("--graph-expand", default=0, type=int)
@click.option("--contract-type", "label", default=None, help="Filter by contract_type metadata")
@click.option("--date-from", default=None)
@click.option("--date-to", default=None)
@click.option("--rerank", is_flag=True)
@click.option("--agentic/--no-agentic", default=False)
@click.option("--api-base", default=None)
@click.option("--api-key", default=None)
@click.option("--model", default=LITELLM_MODEL)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def search_cmd(query, mode, limit, expand, graph_expand, label,
               date_from, date_to, rerank, agentic, api_base, api_key,
               model, index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)

        embeddings = load_index(index_path)
        click.echo(f"📊 索引: {embeddings.count()} 个块 | 图: {'✅' if bool(embeddings.graph) else '❌'}")

        if agentic:
            decider = AgenticDecider(api_base=api_base, api_key=api_key, model=model)
            decision = decider.decide(query)
            mode = decision["mode"]
            expand = decision.get("expand", expand)
            graph_expand = decision.get("graph_expand", graph_expand)
            rerank = decision.get("use_rerank", rerank)
            click.echo(f"🤖 Agentic: mode={mode}, expand={expand}, graph={graph_expand}, rerank={rerank}")

        expander = QueryExpander(api_base=api_base, api_key=api_key, model=model) if expand else None
        reranker = Reranker(top_n=DEFAULT_RERANK_TOP_N) if rerank else None
        searcher = Searcher(embeddings, expander=expander, reranker=reranker)

        t0 = time.perf_counter()
        results = searcher.search(
            query=query, mode=mode, limit=limit,
            expand=expand, graph_expand=graph_expand,
            label_filter=label, date_from=date_from, date_to=date_to,
            use_rerank=rerank,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        click.echo(f"🔍 模式: {mode} | 查询: \"{query}\" | expand={'✅' if expand else '❌'} | {len(results)} 条 | {elapsed:.1f}ms\n")
        click.echo(_format_results(results))

    except Exception as e:
        logger.exception("搜索失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("graph-query")
@click.argument("cypher")
@click.option("--limit", default=20, type=int)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def graph_query_cmd(cypher, limit, index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        if not embeddings.graph:
            click.echo("❌ 图未启用", err=True)
            sys.exit(1)
        searcher = Searcher(embeddings)
        results = searcher.graph_query(cypher, limit)
        if results:
            click.echo(tabulate(results, headers="keys", tablefmt="simple_grid"))
        else:
            click.echo("无结果。")
    except Exception as e:
        logger.exception("图查询失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("benchmark")
@click.option("--runs", default=20, type=int)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def benchmark_cmd(runs, index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        searcher = Searcher(embeddings)
        queries = ["诈骗", "合作方案", "project report", "urgent", "投资"]
        click.echo(f"📊 {embeddings.count()} 个块, 每个查询 {runs} 次\n")
        all_latencies: List[float] = []
        rows = []
        for q in queries:
            latencies: List[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = searcher.search(q, mode="hybrid", limit=10, expand=False)
                latencies.append((time.perf_counter() - t0) * 1000)
            all_latencies.extend(latencies)
            avg = statistics.mean(latencies)
            p50 = statistics.median(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            p99 = sorted(latencies)[int(len(latencies) * 0.99)]
            rows.append([q, f"{avg:.1f}", f"{p50:.1f}", f"{p95:.1f}", f"{p99:.1f}"])
        click.echo(tabulate(rows, headers=["Query", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)"], tablefmt="simple_grid"))
        if all_latencies:
            sorted_all = sorted(all_latencies)
            click.echo(f"\n📈 总体: Avg={statistics.mean(all_latencies):.1f}ms | P50={statistics.median(all_latencies):.1f}ms | P95={sorted_all[int(len(sorted_all)*0.95)]:.1f}ms | P99={sorted_all[int(len(sorted_all)*0.99)]:.1f}ms")
    except Exception as e:
        logger.exception("基准测试失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("synthetic-benchmark")
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def synthetic_benchmark_cmd(index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        searcher = Searcher(embeddings)
        click.echo("🧪 合成测试查询基准\n")
        rows = []
        hit_count = 0
        for test in SYNTHETIC_TEST_QUERIES:
            results = searcher.search(test["query"], mode="hybrid", limit=10, expand=False)
            found_contract_types = set()
            found_counterparties = set()
            for r in results:
                meta = r.get("metadata", {})
                contract_type = meta.get("contract_type")
                counterparty = meta.get("counterparty_name")
                if contract_type not in (None, ""):
                    found_contract_types.add(contract_type)
                if counterparty:
                    found_counterparties.add(counterparty)
            matched_type = bool(set(test["expected_contract_types"]) & found_contract_types)
            matched_counterparty = True
            if test["expected_counterparties"]:
                matched_counterparty = any(any(exp in cp for cp in found_counterparties) for exp in test["expected_counterparties"])
            matched = matched_type and matched_counterparty
            if matched:
                hit_count += 1
            rows.append([
                "✅" if matched else "❌",
                test["query"][:25],
                test["description"][:30],
                ", ".join(map(str, test["expected_contract_types"])),
                ", ".join(sorted(map(str, found_contract_types))[:4]) or "-",
                ", ".join(sorted(found_counterparties)[:3]) or "-",
            ])
        click.echo(tabulate(rows, headers=["", "Query", "Description", "Expected Contract Types", "Found Contract Types", "Found Counterparties"], tablefmt="simple_grid"))
        total = len(SYNTHETIC_TEST_QUERIES)
        click.echo(f"\n📊 召回率: {hit_count}/{total} = {hit_count/total*100:.1f}%")
    except Exception as e:
        logger.exception("合成基准失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)



# ==================================================================
# CrossTableAgent - real tool wiring
# ==================================================================

def build_contract_tool(embeddings, searcher=None, planner=None):
    """Build the unified contract search tool (tickets 01-02).

    One tool for both contract and risk search: the service planner extracts
    risk filters/rank hints internally (no mode gate). The returned callable
    satisfies the agent tool contract: tool(query, filters) -> observation.
    The full ContractRow set is stashed in the result store for the UI; the
    LLM only ever sees the budget-capped observation text.
    """
    if planner is None:
        planner = UnifiedQueryPlanner()
    if searcher is not None:
        service = ContractSearchService(searcher=searcher, planner=planner)
    else:
        service = ContractSearchService(embeddings=embeddings, planner=planner)

    def contract_tool(query, filters, rank_by=None):
        # An explicit rank_by from the agent (e.g. "amount" for "largest
        # contracts") overrides the planner's rank hint; otherwise the planner's
        # hint (or relevance) applies via the service default.
        rows = service.search(query, filters=filters, rank_by=rank_by)
        stash_results(
            rows,
            query=query,
            filters=filters,
            rank_by=rank_by or (service.last_plan or {}).get("rank_hint") or "relevance",
            observation_count=min(len(rows), OBSERVATION_ROW_BUDGET),
        )
        return format_contract_observation(rows)

    return contract_tool


def build_risk_tool(embeddings, api_base=None, api_key=None, model=None):
    """Deprecated: the risk tool is retired as a separate LLM tool (ticket 02).

    Kept as a back-compat shim for legacy callers: returns a query-only
    adapter over the unified contract tool. Risk filters and risk ranking
    come from the shared planner inside the service; there is no mode gate.
    """
    unified = build_contract_tool(embeddings)

    def risk_tool(query):
        return unified(query, {})

    return risk_tool


def build_where_tool(embeddings=None, searcher=None, service=None, llm_client=None):
    """Build the contracts_where tool (ticket 05, Phase 2).

    Exact structured retrieval: natural-language condition -> validated SQL
    over the sections table (rule-based first, LLM text-to-SQL fallback,
    index-scan last resort). Results are the same ContractRow set stashed in
    the same result store — one table, one detail view regardless of which
    tool ran. Bare "list all" conditions return every contract.
    """
    if service is None:
        if searcher is not None:
            service = ContractSearchService(searcher=searcher)
        else:
            service = ContractSearchService(embeddings=embeddings)

    def contracts_where(condition):
        rows = service.search_where(condition, llm_client=llm_client)
        stash_results(
            rows,
            query=condition or "",
            filters={},
            rank_by="relevance",
            observation_count=min(len(rows), OBSERVATION_ROW_BUDGET),
        )
        return format_contract_observation(rows)

    return contracts_where


def build_aggregate_tool(embeddings=None, searcher=None, service=None):
    """Build the contracts_aggregate tool (SQL-side aggregate, spec agent_aggregate_rank_tools).

    The agent passes (metric, group_by, condition); the database computes the
    aggregate over the FULL matching contract set (per-contract deduped), so
    totals are correct — never a LIMIT-capped row fetch summed in Python.
    Returns a rendered text table string. Unlike contracts_where, there is no
    row set to stash (the output IS the summary table), so nothing is added to
    the result store.
    """
    if service is None:
        if searcher is not None:
            service = ContractSearchService(searcher=searcher)
        else:
            service = ContractSearchService(embeddings=embeddings)

    def contracts_aggregate(metric, group_by="", condition=""):
        return service.aggregate(metric, group_by, condition)

    return contracts_aggregate


def build_detail_tool(embeddings=None, searcher=None, service=None):
    """Build the contract_detail tool (single-contract drill-down).

    Returns a readable field summary string for one contract reference; the
    contract's chunks are merged so no field is lost to chunking. No result
    stash — the output is the summary text itself.
    """
    if service is None:
        if searcher is not None:
            service = ContractSearchService(searcher=searcher)
        else:
            service = ContractSearchService(embeddings=embeddings)

    def contract_detail(ref):
        return service.contract_detail(ref)

    return contract_detail


def build_compare_tool(embeddings=None, searcher=None, service=None):
    """Build the contracts_compare tool (side-by-side comparison across refs).

    Returns an aligned text table comparing the given contract references field
    by field. No result stash — the output is the comparison table itself.
    """
    if service is None:
        if searcher is not None:
            service = ContractSearchService(searcher=searcher)
        else:
            service = ContractSearchService(embeddings=embeddings)

    def contracts_compare(refs):
        return service.contracts_compare(refs)

    return contracts_compare


def _load_full_sections(searcher):
    """Load all sections from the index DB as a DataFrame with decoded labels."""
    import json
    from apps.search._core import _clean_text_from_enriched

    db = getattr(getattr(searcher, 'embeddings', None), 'database', None)
    if db is None:
        return pd.DataFrame()
    conn = db.connection
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, tags FROM sections')
        rows = cursor.fetchall()
    finally:
        cursor.close()

    records = []
    for row in rows:
        doc_id = row[0]
        meta = {}
        try:
            meta = json.loads(row[1]) if row[1] else {}
        except Exception:
            pass
        decoded = meta.get('decoded_fields') or {}
        rec = {'id': doc_id}
        for field, pair in decoded.items():
            rec[field] = (pair or {}).get('label')
        rec['_contract_id'] = meta.get('contract_id')
        rec['_ref_no'] = meta.get('ref_no')
        rec['_title'] = meta.get('title')
        rec['_counterparty_name'] = meta.get('counterparty_name')
        rec['_department'] = meta.get('department')
        rec['_amount'] = meta.get('amount')
        rec['_contract_type'] = meta.get('contract_type')
        records.append(rec)
    return pd.DataFrame(records)


def _format_risk_results(risk_results: pd.DataFrame) -> str:
    """Format risk search results into an LLM-readable observation string."""
    if risk_results.empty:
        return ''
    out = []
    for _, row in risk_results.iterrows():
        ref = row.get('_ref_no') or row.get('id') or '?'
        party = row.get('_counterparty_name') or row.get('_title') or '?'
        score = row.get('risk_score', 0)
        severity = row.get('risk_severity', 'unknown')
        signals = row.get('matched_signals', [])
        explanation = row.get('risk_explanation', '')
        signals_str = '; '.join(signals[:5]) if signals else 'no signals'
        out.append(f'[ref={ref} | {party}] risk_score={score} severity={severity} signals: {signals_str}')
    return chr(10).join(out)




@cli.command("agent")
@click.argument("query")
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
@click.option("--show-steps", is_flag=True, help="Print the agent process timeline.")
def agent_cmd(query, index_path, show_steps):
    """Run the CrossTableAgent (contract-domain agentic search)."""
    from apps.search import CrossTableAgent
    try:
        if not os.path.exists(index_path):
            click.echo("❌ 索引不存在: %s" % index_path, err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        agent = CrossTableAgent(
            contract_tool=build_contract_tool(embeddings),
        )
        result = agent.process(query)
        if show_steps:
            click.echo("-- Process --")
            for s in result["steps"]:
                click.echo("  %s %s: %s" % (s["icon"], s["label"], s["detail"]))
            click.echo("-- intent=%s tool=%s fallback=%s --" % (
                result["intent"], result["tool"], result["fallback"]))
        click.echo(result["output"])
        if not result["success"]:
            sys.exit(1)
    except Exception as e:
        logger.exception("Agent failed")
        click.echo("❌ %s" % e, err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
