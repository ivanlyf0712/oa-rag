#!/usr/bin/env python3
"""
CorpChat Search CLI — 精炼版 Onyx 风格搜索框架
================================================
基于 analysis_report.md 的设计蓝图构建，整合 txtai 高性能索引。

核心机制 (对照 Onyx):
  1. 索引管道 (§2.1-§2.3):
     - 句子级分块 (chonkie SentenceChunker, chunk_size=256 tokens)
     - 块丰富化 (标题 + 内容 + 元数据 → 嵌入文本)
     - 双重索引: 丰富化文本 (语义) + 原始文本 (关键词)
  2. 搜索管道 (§2.5-§2.8):
     - 多查询扩展: 语义重写 + 关键词提取 (LiteLLM)
     - 加权 RRF 融合 (原始 0.5 / 语义 1.3 / 关键词 1.0, k=50)
     - 混合搜索 (txtai hybrid: BM25 + 向量)
     - 图增强 (邻居一跳, 仅对 top-3 扩展, 折扣得分)
     - 交叉编码器重排序 (rerank_top_n=20)
  3. Agentic 决策 (§2.7):
     - 规则优先 + 复杂度分析 + LLM 回退
     - 决定: mode, expand, graph_expand, use_rerank

使用方法:
  python apps/search_cli.py build [--force] [--graph-mode auto|llm|off]
  python apps/search_cli.py search "合同 违约责任" --mode hybrid --expand
  python apps/search_cli.py benchmark --runs 20

依赖:
  pip install txtai pymysql click tabulate chonkie sentence-transformers jieba
  可选: python-dotenv (环境变量)
"""

import os
import sys
import json
import re
import time
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import txtai
from tabulate import tabulate

# ── 中文分词 (jieba) ─────────────────────────────────────────────
try:
    import jieba
    jieba.setLogLevel(20)  # silence jieba's build-dict logging
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

# ── 环境变量 (.env) ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except ImportError:
    pass

# ── 路径 & 配置 ──────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.db import fetch_contracts


# ── 日志 ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("oa-search")

# ═══════════════════════════════════════════════════════════════════
# 1. 配置与常量
#    参考 analysis_report.md §2.2 (分块), §2.8 (RRF), §2.7 (权重)
# ═══════════════════════════════════════════════════════════════════

# 嵌入模型: 本地缓存优先 (默认 bge-m3 — 中文检索能力)
_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
_LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "bge-m3")
if os.path.isdir(_LOCAL_MODEL_PATH):
    _EMBED_MODEL = _LOCAL_MODEL_PATH

# 索引路径
DEFAULT_INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(os.path.dirname(__file__), "search_index"))

# 分块参数 (§2.2)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))

# 搜索参数 (§2.5, §2.8)
DEFAULT_HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
RRF_K_VALUE = 50
MAX_SEARCH_LIMIT = 100
# 重排序模型: 中文能力 (BAAI/bge-reranker-base — 中文/多语言交叉编码器)
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_RERANK_TOP_N = 20

# 查询权重 (§2.7 的 constants.py)
ORIGINAL_QUERY_WEIGHT = 0.5
LLM_SEMANTIC_QUERY_WEIGHT = 1.3
LLM_KEYWORD_QUERY_WEIGHT = 1.0

# ── LLM 配置 (LiteLLM proxy — 唯一 provider) ─────────────────────────
# 密钥必须从环境变量提供, 不硬编码。所有 LLM 调用统一经由
# apps/search/litellm_client.py 的 LiteLLMClient 发出。
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.dchbi.app").rstrip("/")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "dseek-v4-flash")
# 单次请求超时 (秒): 代理峰值延迟实测可达 ~40s, 默认 90s 留足余量。
# 45s default (was 90): fail faster when the shared proxy stalls, so
# the router fallback kicks in sooner. Override via LITELLM_TIMEOUT env.
LITELLM_TIMEOUT = int(os.getenv("LITELLM_TIMEOUT", "45"))
# 单次响应 max_tokens 上限 (决策与合成共用)。
LITELLM_MAX_TOKENS = int(os.getenv("LITELLM_MAX_TOKENS", "2048"))

# 富文本 Metadata 格式标记 (已弃用 — 元数据现存于 sections.tags 列)
_METADATA_MARKER = "\n---\nMetadata: "


# ═══════════════════════════════════════════════════════════════════


def _matches_filter_set(value: Any, raw_filter: Optional[str]) -> bool:
    if not raw_filter:
        return True
    wanted = {part.strip() for part in str(raw_filter).split(",") if part.strip()}
    return str(value) in wanted

# 辅助: 从 enriched text 中提取干净内容
#     metadata 现从 sections.tags 列按 id 查询 (见 _fetch_one_doc),
#     不再从文本字符串中反向解析。
# ═══════════════════════════════════════════════════════════════════

def _clean_text_from_enriched(text: str) -> str:
    """
    从 enriched text 中提取干净的内容文本 (去掉 title 前缀和 metadata 后缀)。
    
    返回: 去除了标题行和 Metadata 部分的原始消息内容。
    """
    # 去掉 Metadata 后缀 (兼容旧索引)
    if _METADATA_MARKER in text:
        text = text.split(_METADATA_MARKER)[0]
    # 去掉 title 前缀 (第一行 "---" 之前的内容和 "---" 分隔符)
    parts = text.split("\n---\n", 1)
    if len(parts) > 1:
        return parts[1]
    return text


def _segment(text: str) -> str:
    """
    使用 jieba 对中文文本进行分词, 以空格连接。

    使 txtai 默认的 Unicode 分词器能按 jieba 的词语边界切分中文,
    从而让 BM25 能匹配未加空格的中文短语 (如 投資美國債券跟藍籌股)。
    索引与查询两侧使用同一分词器, 保证一致性。
    """
    if not text:
        return text
    if _JIEBA_AVAILABLE:
        return " ".join(jieba.cut_for_search(text))
    return text


def _compute_contract_relationships(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    从合同分块元数据计算结构关系, 返回 {chunk_id: [relationships]}.

    关系类型: same_contract, same_counterparty, same_department, same_contract_type.
    """
    metas = {chunk["id"]: chunk.get("metadata", {}) for chunk in chunks}
    relationships: Dict[str, List[Dict]] = {cid: [] for cid in metas}
    ids = list(metas.keys())

    for i, a_id in enumerate(ids):
        a = metas[a_id]
        for b_id in ids[i + 1:]:
            b = metas[b_id]
            rels: set = set()

            if a.get("contract_id") is not None and a["contract_id"] == b.get("contract_id"):
                rels.add("same_contract")

            if a.get("counterparty_name") and a["counterparty_name"] == b.get("counterparty_name"):
                rels.add("same_counterparty")

            if a.get("department") and a["department"] == b.get("department"):
                rels.add("same_department")

            if a.get("contract_type") and a["contract_type"] == b.get("contract_type"):
                rels.add("same_contract_type")

            for rel in rels:
                relationships[a_id].append({"id": b_id, "relation": rel})
                relationships[b_id].append({"id": a_id, "relation": rel})

    return relationships


# ═══════════════════════════════════════════════════════════════════
# 2. 索引构建器 (IndexBuilder)
#    参考 §2.2 (分块策略) 和 §2.3 (块丰富化)
# ═══════════════════════════════════════════════════════════════════

class IndexBuilder:
    """构建带分块、丰富化和元数据的合同混合搜索索引。"""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 fetch_limit: int = 10000):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fetch_limit = fetch_limit

    # ── 数据读取 ──────────────────────────────────────────────
    def _fetch_contracts(self) -> List[Dict]:
        """从 MySQL 读取规范化合同记录 (core.db.fetch_contracts)。"""
        return fetch_contracts(limit=self.fetch_limit)

    # ── 可索引文本面 ──────────────────────────────────────────
    def _indexable_text(self, contract: Dict) -> str:
        """
        组合合同的可检索文本面: 标题 + 相对人 + 产品服务 + 正文 + 解码后的状态字段。
        元数据 (金额/部门/日期等) 走 tags 列; 但解码后的布尔/风险状态
        (search_context: 如 "Over5M: yes") 拼入正文, 让编码器和 BM25
        能按业务含义检索 (§raw-first 反规范化)。
        """
        parts: List[str] = []
        for key in ("title", "counterparty_name", "product_services", "content"):
            value = contract.get(key)
            if value and str(value).strip():
                parts.append(str(value).strip())
        # 解码后的状态字段行 ("Field: yes/no/na") — 让 0/1/2 代码可被含义检索
        search_context = contract.get("search_context")
        if search_context:
            parts.append("\n".join(str(line) for line in search_context))
        # 去重, 保持顺序
        seen: set = set()
        unique: List[str] = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique.append(part)
        return "\n\n".join(unique)

    # ── 分块 (§2.2) ────────────────────────────────────────────
    def _chunk_contract(self, contract: Dict) -> List[Dict]:
        """将单条合同拆分为句子级块。"""
        content = self._indexable_text(contract)
        if not content:
            return []
        chunks_text: List[str] = []

        try:
            from chonkie import SentenceChunker

            def _token_counter(text: str) -> int:
                chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
                other_chars = len(text) - chinese_chars
                return int(chinese_chars / 2 + other_chars / 4)

            chunker = SentenceChunker(
                tokenizer_or_token_counter=_token_counter,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                return_type="texts",
            )
            chunks_text = chunker.chunk(content)
        except (ImportError, Exception) as e:
            logger.debug(f"chonkie 不可用 ({e}), 使用 fallback 分块")
            import re
            sentences = re.split(r'(?<=[.!?。！？])\s*', content)
            current = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len < self.chunk_size * 4:
                    current.append(sent)
                    current_len += sent_len
                else:
                    if current:
                        chunks_text.append(" ".join(current).strip())
                    current = [sent]
                    current_len = sent_len
            if current:
                chunks_text.append(" ".join(current).strip())

        if not chunks_text:
            chunks_text = [content]

        base_id = f"contract_{contract.get('id')}"
        chunks: List[Dict] = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{base_id}__chunk{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "contract_id": contract.get("id"),
                    "request_id": contract.get("request_id"),
                    "ref_no": contract.get("ref_no"),
                    "title": contract.get("title"),
                    "counterparty_name": contract.get("counterparty_name"),
                    "product_services": contract.get("product_services"),
                    "department": contract.get("department"),
                    "department_id": contract.get("department_id"),
                    "amount": contract.get("amount"),
                    "amount_label": contract.get("amount_label"),
                    "contract_start_date": contract.get("contract_start_date"),
                    "contract_end_date": contract.get("contract_end_date"),
                    "requested_date": contract.get("requested_date"),
                    "status": contract.get("status"),
                    "status_label": contract.get("status_label"),
                    "contract_type": contract.get("contract_type"),
                    "legal_approval": contract.get("legal_approval"),
                    "overruled": contract.get("overruled"),
                    # de-normalized raw-first structures for filter / display
                    "decoded_fields": contract.get("decoded_fields"),
                    "contextual_fields": contract.get("contextual_fields"),
                    # full source record for the detail Raw tab (ticket 04);
                    # present from the next index build onward
                    "raw": contract.get("raw"),
                    "chunk_index": i,
                },
                "title": contract.get("title") or f"Contract #{contract.get('id')}",
            })
        return chunks

    # ── 丰富化 (§2.3) ──────────────────────────────────────────
    def _enrich_chunk(self, chunk: Dict) -> str:
        """
        丰富化: 组合标题 + 内容 → 用于嵌入与匹配的最终文本。

        匹配面 (match surface) 仅包含:
          - 标题: 合同名称 (contract_type)  — 提供消歧上下文
          - 内容: 标题 + 相对人 + 产品服务 + 正文 的拼接
        元数据 (金额, 部门, 日期, 相对人, 状态 等) 不再拼入文本,
        而是作为结构化 tags 存入 sections.tags 列, 用于过滤/展示/LLM 上下文。

        格式: [title]\n---\n[content]
        """
        title = chunk.get("title", "")
        text = chunk["text"]
        # 中文分词: 让 BM25 能匹配未加空格的中文短语
        return f"{title}\n---\n{_segment(text)}"

    # ── 索引构建入口 ──────────────────────────────────────────
    def build(self, force: bool = False, enable_graph: bool = True,
              graph_mode: str = "auto") -> txtai.Embeddings:
        """构建或加载索引。"""
        if os.path.exists(self.index_path) and not force:
            logger.info(f"从 {self.index_path} 加载已有索引 ...")
            embeddings = txtai.Embeddings()
            embeddings.load(self.index_path)
            logger.info(f"已加载 {embeddings.count()} 个块")
            return embeddings

        logger.info("从数据库构建新索引 (含分块+丰富化) ...")
        contracts = self._fetch_contracts()
        if not contracts:
            raise RuntimeError("数据库中没有合同数据")

        all_chunks = []
        for contract in contracts:
            chunks = self._chunk_contract(contract)
            all_chunks.extend(chunks)
        logger.info(f"分块完成: {len(contracts)} 条合同 → {len(all_chunks)} 个块")

        # 计算结构关系 (仅当启用图时)
        relationships: Dict[str, List[Dict]] = {}
        if enable_graph:
            relationships = _compute_contract_relationships(all_chunks)

        docs = []
        for chunk in all_chunks:
            enriched = self._enrich_chunk(chunk)
            tags_json = json.dumps(chunk["metadata"], default=str)
            if enable_graph and relationships:
                docs.append((
                    chunk["id"],
                    {
                        "text": enriched,
                        "relationships": relationships.get(chunk["id"], []),
                    },
                    tags_json,
                ))
            else:
                docs.append((chunk["id"], enriched, tags_json))

        config: Dict = {
            "path": _EMBED_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
            "columns": {"relationships": "relationships"},
        }
        if enable_graph:
            config["graph"] = True

        logger.info(f"模型: {_EMBED_MODEL}")
        logger.info(f"图功能: {'✅' if enable_graph else '❌'} (模式: structural)")

        embeddings = txtai.Embeddings(config)

        t0 = time.perf_counter()
        logger.info(f"索引 {len(docs)} 个文档 ...")
        embeddings.index(docs)
        logger.info(f"索引完成, 耗时 {time.perf_counter()-t0:.2f}s")

        if enable_graph and embeddings.graph:
            logger.info("图构建完成: 纯结构关系 (same_contract / same_counterparty / same_department / same_contract_type)")

        embeddings.save(self.index_path)
        logger.info(f"索引保存至 {self.index_path}")
        return embeddings


# ═══════════════════════════════════════════════════════════════════
# 3. 查询扩展器 (QueryExpander)
#    参考 §2.7 的 LLM 查询扩展
# ═══════════════════════════════════════════════════════════════════

class QueryExpander:
    """使用 LLM 生成语义重写和关键词扩展查询。"""

    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = LITELLM_MODEL):
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self._cache: Dict[str, List[Tuple[str, float]]] = {}

    def _call_llm(self, messages: List[Dict], max_tokens: int = 200) -> str:
        # 经由共享 LiteLLMClient (单一 LLM 入口); 辅助调用用较短超时, 代理
        # 抖动时快速降级到确定性路径而不是长时间阻塞。
        from apps.search.litellm_client import LiteLLMClient
        try:
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key,
                                   model=self.model)
            return client.chat(messages, temperature=0.1, max_tokens=max_tokens,
                               timeout=30).strip()
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return ""

    def _semantic_rephrase(self, query: str) -> Optional[str]:
        system_msg = (
            "You reformulate user queries into standalone semantic search queries. "
            "Output ONLY the reformulated query, no extra text."
        )
        user_msg = (
            f"Rewrite this query into a standalone semantic search query. "
            f"In most cases keep it identical. Only add missing context or remove "
            f"non-search instructions.\n\nQuery: {query}\n\nSemantic query:"
        )
        result = self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        if result and result != query:
            return result
        return None

    def _keyword_expand(self, query: str) -> List[str]:
        system_msg = (
            "You reformulate user queries into keyword-only queries. "
            "Output ONLY the keywords, one set per line (max 3 lines)."
        )
        user_msg = (
            f"Extract up to 3 keyword-only search queries from the user query. "
            f"Each line should contain one set of keywords.\n\nQuery: {query}\n\nKeywords:"
        )
        result = self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        if not result:
            return []
        keywords = [
            line.strip() for line in result.split("\n")
            if line.strip() and len(line.strip()) > 1
        ]
        return keywords[:3]

    def expand(self, query: str, use_cache: bool = True) -> List[Tuple[str, float]]:
        cache_key = query[:100]
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        results: List[Tuple[str, float]] = [(query, ORIGINAL_QUERY_WEIGHT)]

        try:
            semantic = self._semantic_rephrase(query)
            if semantic and semantic.lower() != query.lower():
                results.append((semantic, LLM_SEMANTIC_QUERY_WEIGHT))
        except Exception as e:
            logger.debug(f"语义重写失败: {e}")

        try:
            kw_queries = self._keyword_expand(query)
            for kw in kw_queries:
                existing = {q.lower() for q, _ in results}
                if kw.lower() not in existing:
                    results.append((kw, LLM_KEYWORD_QUERY_WEIGHT))
        except Exception as e:
            logger.debug(f"关键词扩展失败: {e}")

        self._cache[cache_key] = results
        return results


# ═══════════════════════════════════════════════════════════════════
# 4. 重排序器 (Reranker)
#   参考 §2.6 — 交叉编码器
# ═══════════════════════════════════════════════════════════════════

class Reranker:
    """交叉编码器重排序, 仅对前 rerank_top_n 个结果重排。"""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL,
                 top_n: int = DEFAULT_RERANK_TOP_N):
        self.enabled = False
        self.model = None
        self.model_name = model_name
        self.top_n = top_n
        try:
            from sentence_transformers import CrossEncoder
            self.enabled = True
        except ImportError:
            logger.warning("sentence_transformers 未安装, 重排序已禁用")

    def _ensure_model(self) -> None:
        if self.model is None and self.enabled:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载交叉编码器: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        if not self.enabled or not results:
            return results
        if self.model is None:
            try:
                self._ensure_model()
            except Exception as e:
                logger.warning(f"重排序模型加载失败: {e}")
                return results

        if len(results) <= self.top_n:
            to_rerank = results
            rest = []
        else:
            to_rerank = results[:self.top_n]
            rest = results[self.top_n:]

        pairs = [(query, item.get("text", "")) for item in to_rerank]
        try:
            scores = self.model.predict(pairs)
            for i, score in enumerate(scores):
                to_rerank[i]["rerank_score"] = float(score)
                # Keep original score (RRF or hybrid) for display; use rerank_score for sorting only
            to_rerank.sort(key=lambda x: float(x.get("rerank_score", 0)), reverse=True)
        except Exception as e:
            logger.warning(f"重排序失败: {e}")

        return to_rerank + rest


# ═══════════════════════════════════════════════════════════════════
# 5. 搜索器 (Searcher)
#    实现混合搜索 + RRF 融合 + 图扩展 + 重排序
#    参考 §2.5 (混合搜索), §2.6 (重排序), §2.8 (RRF 融合)
# ═══════════════════════════════════════════════════════════════════

class Searcher:
    """
    多模式搜索器: keyword / semantic / hybrid + 图增强 + 重排序。

    可直接被 app.py 导入使用:
      from apps.search import Searcher, load_index
      searcher = Searcher(load_index())
    """

    def __init__(self, embeddings: txtai.Embeddings,
                 expander: Optional[QueryExpander] = None,
                 reranker: Optional[Reranker] = None):
        self.embeddings = embeddings
        self.expander = expander
        self.reranker = reranker

    # ── 加权 RRF 融合 (§2.8) ─────────────────────────────────
    @staticmethod
    def _weighted_rrf_fusion(
        all_results: List[Tuple[List[Tuple[str, float]], float]],
        k: int = RRF_K_VALUE
    ) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        source_rank: Dict[str, int] = {}
        source_idx: Dict[str, int] = {}

        for q_idx, (result_list, weight) in enumerate(all_results):
            for rank, (doc_id, _) in enumerate(result_list, start=1):
                if not doc_id:
                    continue
                rrf_score = weight / (k + rank)
                scores[doc_id] += rrf_score
                if doc_id not in source_rank:
                    source_rank[doc_id] = rank
                    source_idx[doc_id] = q_idx

        sorted_ids = sorted(
            scores.keys(),
            key=lambda did: (-scores[did], source_rank.get(did, 999), source_idx.get(did, 999))
        )
        return [(did, scores[did]) for did in sorted_ids]

    # ── 图扩展 (纯结构, 直接 backend API) ──────────────────────
    def _graph_expand(self, results: List[Dict], max_expand: int = 3,
                       hop_discount: float = 0.8, limit: int = 20,
                       query: str = "", label_filter: Optional[str] = None,
                       date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[Dict]:
        """
        从 base 结果出发, 遍历 4 种 traversal-eligible 结构边,
        将邻居追加到 base 结果下方 (不重排 base).

        score = parent_score × hop_discount × neighbor_query_relevance
        其中 neighbor_query_relevance 通过 already-loaded 索引对邻居文本做一次
        轻量 hybrid search 获得, 实现 query-consistency gate.
        """
        graph = self.embeddings.graph
        if not graph or not results:
            return results

        # Build doc-id -> node-key map once
        id_to_key = {}
        for key, attrs in graph.scan(data=True):
            id_to_key[attrs["id"]] = key

        # Only traverse these 4 relation types; same_label is recorded but never traversed
        TRAVERSAL_RELATIONS = {
            "same_contract", "same_counterparty",
            "same_department", "same_contract_type",
        }

        # Query-consistency gate: run the query search ONCE, build id -> score map.
        # Avoids an N+1 full search per neighbor.
        query_scores: Dict[str, float] = {}
        if query:
            try:
                q_raw = self.embeddings.search(_segment(query), limit=MAX_SEARCH_LIMIT)
                for item in q_raw:
                    parsed = self._parse_txtai_result(item)
                    if parsed:
                        query_scores[parsed["id"]] = parsed.get("score", 0.0)
            except Exception:
                query_scores = {}

        def _passes_filters(doc: Dict) -> bool:
            meta = doc.get("metadata", {})
            if not _matches_filter_set(meta.get("contract_type"), label_filter):
                return False
            date_value = meta.get("requested_date", "")
            if date_from and date_value and str(date_value) < date_from:
                return False
            if date_to and date_value and str(date_value) > date_to:
                return False
            return True

        expanded_ids = {r["id"] for r in results}
        expanded = list(results)

        seeds = results[:min(max_expand, len(results))]
        for r in seeds:
            seed_id = r["id"]
            seed_key = id_to_key.get(seed_id)
            if seed_key is None:
                continue

            neighbors = graph.edges(seed_key)
            if not neighbors:
                continue

            for neighbor_key, edge_attrs in neighbors.items():
                relation = edge_attrs.get("relation", "")
                if relation not in TRAVERSAL_RELATIONS:
                    continue

                # Resolve neighbor doc id from node attributes
                neighbor_attrs = graph.node(neighbor_key)
                if not neighbor_attrs:
                    continue
                neighbor_id = neighbor_attrs.get("id")
                if not neighbor_id or neighbor_id in expanded_ids:
                    continue

                # Query-consistency gate: look up the precomputed relevance score.
                neighbor_query_relevance = query_scores.get(neighbor_id, 0.0)

                final_score = r.get("score", 0.0) * hop_discount * neighbor_query_relevance
                if final_score <= 0.0:
                    # Irrelevant neighbor: balanced out, do not surface
                    continue

                neighbor_doc = self._fetch_one_doc(neighbor_id)
                if not neighbor_doc:
                    continue

                # Re-apply label/date filters
                if not _passes_filters(neighbor_doc):
                    continue

                expanded_ids.add(neighbor_id)
                expanded.append({
                    "id": neighbor_id,
                    "text": neighbor_doc.get("text", ""),
                    "score": final_score,
                    "metadata": {
                        **neighbor_doc.get("metadata", {}),
                        "_graph_relation": relation,
                        "_from_node": seed_id[:30],
                    },
                })

        # Append-only: base order preserved; expanded docs sorted below by score
        base_part = [d for d in expanded if not d.get("metadata", {}).get("_graph_relation")]
        extra_part = [d for d in expanded if d.get("metadata", {}).get("_graph_relation")]
        extra_part.sort(key=lambda x: x.get("score", 0), reverse=True)
        return (base_part + extra_part)[:limit]

    # ── 从 txtai 获取单个文档并提取 metadata ───────────────
    @staticmethod
    def _parse_txtai_result(item: Any) -> Optional[Dict]:
        """
        将 txtai 搜索结果统一解析为 {id, text, score, metadata} 格式。
        
        txtai 返回格式:
          - dict: {id, text, score, tags(optional)}
          - tuple: (id, text, tags_json, score)
        
        注意: txtai 在 content=True 且 objects=True 的配置下, search() 返回的
        dict 中不包含 tags 字段。metadata 通过 _fetch_one_doc 从 sections.tags
        列按 id 查询获取 (见 _fetch_one_doc)。
        """
        doc_id = ""
        text = ""
        score = 0.0

        if isinstance(item, dict):
            doc_id = item.get("id", "")
            text = item.get("text", "")
            score = item.get("score", 0.0)
        elif isinstance(item, tuple) and len(item) >= 4:
            doc_id = item[0]
            text = item[1]
            score = item[3]
        else:
            return None

        if not doc_id:
            return None

        return {
            "id": doc_id,
            "text": text,
            "score": score,
            "metadata": {},
        }

    def _fetch_one_doc(self, doc_id: str) -> Optional[Dict]:
        """
        通过 doc_id 从索引的 sections 表取出文档文本与结构化 tags。

        修复: 旧实现用 embeddings.search(f"id:{doc_id}") 做文本搜索,
        那是一次错误的 BM25 查询, 会返回错误文档。这里改为按 id 直接
        查询 SQLite sections 表, 同时取回 tags 元数据。
        """
        try:
            db = self.embeddings.database
            if db is None:
                return None
            conn = db.connection
            cur = conn.cursor()
            cur.execute("SELECT text, tags FROM sections WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if not row:
                return None
            text, tags_json = row
            metadata = {}
            if tags_json:
                try:
                    metadata = json.loads(tags_json) if isinstance(tags_json, str) else dict(tags_json)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            return {
                "id": doc_id,
                "text": text,
                "score": 0.0,
                "metadata": metadata,
            }
        except Exception as e:
            logger.debug(f"按 id 获取文档失败 {doc_id}: {e}")
            return None


    def _search_deterministic_filters(self, filters: Dict[str, Any], limit: int = 10) -> List[Dict]:
        db = self.embeddings.database
        if db is None or not filters:
            return []
        conn = db.connection
        cur = conn.cursor()
        cur.execute("SELECT id, text, tags FROM sections")
        rows = cur.fetchall()
        out: List[Dict] = []
        for row in rows:
            if len(row) == 3:
                doc_id, text, tags_json = row
            else:
                doc_id, text, tags_json = row[0], row[1], row[-1]
            try:
                meta = json.loads(tags_json) if tags_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            matched = True
            for clause in filters:
                field = clause.get("field")
                value = clause.get("value")
                if field is None:
                    continue
                if not _matches_filter_set(meta.get(field), value):
                    matched = False
                    break
            if matched:
                out.append({"id": doc_id, "text": text, "score": 0.0, "metadata": meta})
            if len(out) >= limit:
                break
        return out

    def _search_exact_ref(self, query: str, label_filter=None, date_from=None, date_to=None) -> List[Dict]:
        """Fetch the exact ref_no document deterministically when the query is ref-like."""
        db = self.embeddings.database
        if db is None:
            return []
        conn = db.connection
        cur = conn.cursor()
        cur.execute("SELECT id, tags FROM sections")
        rows = cur.fetchall()
        q = query.strip().upper()
        out: List[Dict] = []
        for row in rows:
            if len(row) == 2:
                doc_id, tags_json = row
            else:
                doc_id, tags_json = row[0], row[-1]
            try:
                meta = json.loads(tags_json) if tags_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if str(meta.get("ref_no") or "").strip().upper() != q:
                continue
            if label_filter and not _matches_filter_set(meta.get("contract_type"), label_filter):
                continue
            date_value = meta.get("requested_date", "")
            if date_from and date_value and str(date_value) < date_from:
                continue
            if date_to and date_value and str(date_value) > date_to:
                continue
            doc = self._fetch_one_doc(doc_id)
            if doc:
                out.append(doc)
                break
        return out

    # ── 搜索主入口 ──────────────────────────────────────────
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        expand: bool = True,
        graph_expand: int = 0,
        label_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[Dict]:
        """
        执行搜索 (默认启用全链路 Onyx 风格搜索)。
        
        全链路 = LLM 查询扩展 + 混合搜索 + RRF 融合 + 交叉编码器重排序。
        
        当 expand=True 且 self.expander 可用时:
          - 生成语义重写 + 关键词扩展查询
          - 每条查询独立执行 txtai hybrid search
          - 加权 RRF 融合所有结果
          - Reranker 对 RRF 融合后的 top-N 重排序
        返回 RRF 分数 (小数值)。
        
        当 expand=False 或 expander 不可用时:
          - 直接执行 txtai hybrid search
          - 分数 0~1 (原生向量+BM25)
        
        use_rerank=True 且 reranker 可用时:
          - 对最终 top-20 结果用交叉编码器重排序
        """
        weight_map = {
            "keyword": (0.0, 1.0),
            "semantic": (1.0, 0.0),
            "hybrid": None,
        }
        weights = weight_map.get(mode, None)

        def _filter(item: Dict) -> bool:
            meta = item.get("metadata", {})
            if not _matches_filter_set(meta.get("contract_type"), label_filter):
                return False
            date_value = meta.get("requested_date", "")
            if date_from and date_value and str(date_value) < date_from:
                return False
            if date_to and date_value and str(date_value) > date_to:
                return False
            return True

        # 中文分词: 查询与索引使用同一 jieba 分词, 保证 BM25 匹配一致
        segmented_query = _segment(query)

        # Deterministic exact/filter shortcut for structured filters.
        if isinstance(label_filter, dict) and label_filter:
            filtered = self._search_deterministic_filters(label_filter, limit=limit)
            if filtered:
                return filtered[:limit]

        # ── 路径 A: 直接 txtai 搜索 ──
        if not expand or not self.expander:
            raw = self.embeddings.search(segmented_query, limit=min(limit * 3, MAX_SEARCH_LIMIT), weights=weights)
            output = []
            for item in raw:
                parsed = self._parse_txtai_result(item)
                if parsed:
                    # 按 id 取回结构化 tags 元数据 (过滤/展示用)
                    doc = self._fetch_one_doc(parsed["id"])
                    if doc:
                        parsed["metadata"] = doc["metadata"]
                    if _filter(parsed):
                        output.append(parsed)

            if graph_expand > 0 and self.embeddings.graph:
                output = self._graph_expand(
                    output[:limit], max_expand=3, limit=limit * 2,
                    query=query, label_filter=label_filter,
                    date_from=date_from, date_to=date_to,
                )
            if use_rerank and self.reranker and self.reranker.enabled:
                output = self.reranker.rerank(query, output)
            # _graph_expand truncates to limit*2 so graph hits can surface below base
            return output

        # ── 路径 B: 多查询扩展 + RRF ──
        queries_with_weights: List[Tuple[str, float]] = [(query, ORIGINAL_QUERY_WEIGHT)]
        try:
            if expand and self.expander:
                queries_with_weights = self.expander.expand(query)
        except Exception as e:
            logger.warning(f"查询扩展失败: {e}")

        all_results: List[Tuple[List[Tuple[str, float]], float]] = []
        for q, q_weight in queries_with_weights:
            raw = self.embeddings.search(_segment(q), limit=min(limit * 3, MAX_SEARCH_LIMIT), weights=weights)
            result_list: List[Tuple[str, float]] = []
            for item in raw:
                parsed = self._parse_txtai_result(item)
                if parsed:
                    result_list.append((parsed["id"], parsed["score"]))
            all_results.append((result_list, q_weight))

        fused = self._weighted_rrf_fusion(all_results)
        output: List[Dict] = []
        seen_ids = set()
        for doc_id, _ in fused:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            doc = self._fetch_one_doc(doc_id)
            if doc and _filter(doc):
                output.append(doc)

        if graph_expand > 0 and self.embeddings.graph:
            output = self._graph_expand(
                output[:limit], max_expand=3, limit=limit * 2,
                query=query, label_filter=label_filter,
                date_from=date_from, date_to=date_to,
            )
        if use_rerank and self.reranker and self.reranker.enabled:
            output = self.reranker.rerank(query, output)

        # _graph_expand truncates to limit*2 so graph hits can surface below base
        return output

    # ── 图查询 ──────────────────────────────────────────────
    def graph_query(self, cypher: str, limit: int = 20) -> List[Dict]:
        if not self.embeddings.graph:
            raise RuntimeError("图未启用")
        results = self.embeddings.graph.search(cypher)
        output = []
        for i, row in enumerate(results[:limit]):
            item = {"row": i + 1}
            if isinstance(row, (tuple, list)):
                for j, val in enumerate(row):
                    item[f"col_{j}"] = str(val)[:80]
            else:
                item["result"] = str(row)[:80]
            output.append(item)
        return output


# ═══════════════════════════════════════════════════════════════════
# 7. 便捷加载函数
# ═══════════════════════════════════════════════════════════════════

def load_index(index_path: Optional[str] = None) -> txtai.Embeddings:
    path = index_path or DEFAULT_INDEX_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引不存在: {path}。请先运行 python search.py build")
    embeddings = txtai.Embeddings()
    embeddings.load(path)
    if embeddings.graph is not None:
        try:
            import grandcypher  # noqa: F401
        except ImportError:
            # txtai's search() calls graph.isquery() which requires the
            # optional GrandCypher extra. Without it every search crashes.
            # oa-rag never issues graph queries, so strip the graph.
            logger.warning("GrandCypher not installed; disabling index graph "
                           "(graph queries unavailable, vector search unaffected)")
            embeddings.graph = None
    return embeddings


# ═══════════════════════════════════════════════════════════════════
# 7. Agentic 决策器
# ═══════════════════════════════════════════════════════════════════

class AgenticDecider:
    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = LITELLM_MODEL):
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self._mode_cache: Dict[str, str] = {}

    def decide(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        q_len = len(query.split())
        decision = {"mode": "hybrid", "expand": True, "graph_expand": 0, "use_rerank": True}
        question_kws = {"谁", "什么", "何时", "where", "when", "who", "哪个", "如何"}
        similarity_kws = {"类似", "相关", "similar", "related", "like"}
        if any(kw in q_lower for kw in question_kws):
            decision["mode"] = "keyword"; decision["expand"] = False
        elif any(kw in q_lower for kw in similarity_kws):
            decision["mode"] = "semantic"; decision["expand"] = True
        if q_len > 5 or any(c in q_lower for c in ["和", "以及", "对比", "比较", "vs"]):
            decision["graph_expand"] = 1; decision["use_rerank"] = True
        elif q_len <= 2:
            decision["use_rerank"] = False
        try:
            mode_from_llm = self._llm_decide_mode(query)
            if mode_from_llm:
                decision["mode"] = mode_from_llm
        except Exception:
            pass
        return decision

    def _llm_decide_mode(self, query: str) -> Optional[str]:
        # 经由共享 LiteLLMClient; 失败/超时 → None → 调用方用启发式决策。
        from apps.search.litellm_client import LiteLLMClient
        cache_key = query.lower()[:100]
        if cache_key in self._mode_cache:
            return self._mode_cache[cache_key]
        try:
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key,
                                   model=self.model)
            choice = client.chat(
                [{"role": "user",
                  "content": f'For query "{query}", pick ONE: keyword, semantic, hybrid. Reply ONE word.'}],
                temperature=0, max_tokens=10, timeout=30).strip().lower()
            for mode in ["keyword", "semantic", "hybrid"]:
                if mode in choice:
                    self._mode_cache[cache_key] = mode
                    return mode
        except Exception:
            pass
        return None
