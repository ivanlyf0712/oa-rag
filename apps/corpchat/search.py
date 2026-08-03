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
  python apps/corpchat/search.py build [--force] [--graph-mode auto|llm|off]
  python apps/corpchat/search.py search "诈骗" --mode hybrid --expand
  python apps/corpchat/search.py benchmark --runs 20
  python apps/corpchat/search.py validate

依赖:
  pip install txtai psycopg2 click tabulate chonkie sentence-transformers
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

import psycopg2
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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from core.config import DB_CONFIG
except ImportError:
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "dbname": os.getenv("DB_NAME", "invoices"),
        "user": os.getenv("DB_USER", "ocr"),
        "password": os.getenv("DB_PASSWORD", "ocrpass"),
    }


# ── 日志 ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("corpchat-search")

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

# LiteLLM 配置 (密钥必须从环境变量提供, 不硬编码)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.dchbi.app/")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# 富文本 Metadata 格式标记 (已弃用 — 元数据现存于 sections.tags 列)
_METADATA_MARKER = "\n---\nMetadata: "


# ═══════════════════════════════════════════════════════════════════
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


def _compute_structural_relationships(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    从分块元数据计算五个结构关系, 返回 {chunk_id: [relationships]}.

    关系类型: same_conversation, sender_receiver, same_sender, same_company, same_label.
    """
    metas = {chunk["id"]: chunk.get("metadata", {}) for chunk in chunks}
    relationships: Dict[str, List[Dict]] = {cid: [] for cid in metas}
    ids = list(metas.keys())

    for i, a_id in enumerate(ids):
        a = metas[a_id]
        for b_id in ids[i + 1:]:
            b = metas[b_id]
            rels: set = set()

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                rels.add("same_conversation")

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                if a.get("external_userid") == b.get("servicer_userid") or \
                   b.get("external_userid") == a.get("servicer_userid"):
                    rels.add("sender_receiver")

            if a.get("external_userid") and a["external_userid"] == b.get("external_userid"):
                rels.add("same_sender")

            if a.get("company") and a["company"] == b.get("company"):
                rels.add("same_company")

            if a.get("label") and a["label"] == b.get("label"):
                rels.add("same_label")

            for rel in rels:
                relationships[a_id].append({"id": b_id, "relation": rel})
                relationships[b_id].append({"id": a_id, "relation": rel})

    return relationships


# ═══════════════════════════════════════════════════════════════════
# 2. 索引构建器 (IndexBuilder)
#    参考 §2.2 (分块策略) 和 §2.3 (块丰富化)
# ═══════════════════════════════════════════════════════════════════

class IndexBuilder:
    """构建带分块、丰富化和元数据的混合搜索索引。"""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── 数据读取 ──────────────────────────────────────────────
    def _fetch_messages(self) -> List[Dict]:
        """从 PostgreSQL 读取消息及关联联系人信息。"""
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT m.msgid, m.content, m.send_time, m.external_userid,
                   m.servicer_userid, m.label, c.full_name AS customer_name,
                   m.open_kfid, m.origin, c.company
            FROM messages m
            LEFT JOIN contacts c ON m.external_userid = c.userid
            WHERE m.content IS NOT NULL AND m.content != ''
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        messages = []
        for row in rows:
            send_time_raw = row[2]
            if hasattr(send_time_raw, 'isoformat'):
                send_time_str = send_time_raw.isoformat()
            else:
                send_time_str = str(send_time_raw) if send_time_raw else None

            messages.append({
                "msgid": row[0],
                "content": row[1],
                "send_time": send_time_raw,
                "send_time_str": send_time_str,
                "external_userid": row[3],
                "servicer_userid": row[4],
                "label": row[5],
                "customer_name": row[6] or str(row[3]),
                "company": row[9] if len(row) > 9 else None,
                "open_kfid": row[7],
                "origin": row[8],
            })
        return messages

    # ── 分块 (§2.2) ────────────────────────────────────────────
    def _chunk_message(self, msg: Dict) -> List[Dict]:
        """将单条消息拆分为句子级块。"""
        content = msg["content"]
        chunks_text = []

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

        base_id = msg["msgid"]
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{base_id}__chunk{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "msgid": msg["msgid"],
                    "send_time": msg.get("send_time_str"),
                    "external_userid": msg["external_userid"],
                    "servicer_userid": msg["servicer_userid"],
                    "label": msg["label"],
                    "customer_name": msg["customer_name"],
                    "company": msg.get("company"),
                    "open_kfid": msg["open_kfid"],
                    "origin": msg["origin"],
                    "chunk_index": i,
                },
                "title": (
                    f"{msg['customer_name']} ({msg['label'] or 'general'})"
                ),
            })
        return chunks

    # ── 丰富化 (§2.3) ──────────────────────────────────────────
    def _enrich_chunk(self, chunk: Dict) -> str:
        """
        丰富化: 组合标题 + 内容 → 用于嵌入与匹配的最终文本。

        匹配面 (match surface) 仅包含:
          - 标题: customer_name (label)  — 提供消歧上下文
          - 内容: 消息正文
        元数据 (label, 时间, 发送者, 接收者, msgid 等) 不再拼入文本,
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
        messages = self._fetch_messages()
        if not messages:
            raise RuntimeError("数据库中没有消息数据")

        all_chunks = []
        for msg in messages:
            chunks = self._chunk_message(msg)
            all_chunks.extend(chunks)
        logger.info(f"分块完成: {len(messages)} 条消息 → {len(all_chunks)} 个块")

        # 计算结构关系 (仅当启用图时)
        relationships: Dict[str, List[Dict]] = {}
        if enable_graph:
            relationships = _compute_structural_relationships(all_chunks)

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
            logger.info("图构建完成: 纯结构关系 (same_conversation / sender_receiver / same_sender / same_company / same_label)")

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
        import requests
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
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
      from apps.corpchat.search import Searcher, load_index
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
            "same_conversation", "sender_receiver",
            "same_sender", "same_company",
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
            if label_filter and meta.get("label") != label_filter:
                return False
            send_time = meta.get("send_time", "")
            if date_from and send_time and str(send_time) < date_from:
                return False
            if date_to and send_time and str(send_time) > date_to:
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
            if label_filter and meta.get("label") != label_filter:
                return False
            send_time = meta.get("send_time", "")
            if date_from and send_time and str(send_time) < date_from:
                return False
            if date_to and send_time and str(send_time) > date_to:
                return False
            return True

        # 中文分词: 查询与索引使用同一 jieba 分词, 保证 BM25 匹配一致
        segmented_query = _segment(query)

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
        import requests
        cache_key = query.lower()[:100]
        if cache_key in self._mode_cache:
            return self._mode_cache[cache_key]
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user",
                                  "content": f'For query "{query}", pick ONE: keyword, semantic, hybrid. Reply ONE word.'}],
                    "temperature": 0, "max_tokens": 10,
                },
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]["message"]["content"].strip().lower()
            for mode in ["keyword", "semantic", "hybrid"]:
                if mode in choice:
                    self._mode_cache[cache_key] = mode
                    return mode
        except Exception:
            pass
        return None


# ═══════════════════════════════════════════════════════════════════
# 8. 合成测试数据
# ═══════════════════════════════════════════════════════════════════

SYNTHETIC_TEST_QUERIES = [
    {"query": "物流方案报价", "expected_labels": ["product_inquiry"],
     "description": "鴻海陳志明詢問物流系統報價"},
    {"query": "ERP timeout error", "expected_labels": ["tech_support"],
     "description": "長榮張偉強反映 ERP timeout"},
    {"query": "invoice discrepancy", "expected_labels": ["invoice_issue"],
     "description": "勤業廖珮琪核對發票金額差異"},
    {"query": "Microsoft 365 E5 授權價格", "expected_labels": ["software_license"],
     "description": "趨勢謝明宏詢價 M365 E5"},
    {"query": "品質不良率 3%", "expected_labels": ["quality_issue"],
     "description": "鴻準蕭國榮反應零件 3% 不良率"},
    {"query": "聯合促銷活動合作", "expected_labels": ["marketing_campaign"],
     "description": "統一劉德華提聯合促銷"},
    {"query": "Surface Pro 電池續航", "expected_labels": ["warranty_claim"],
     "description": "微軟周怡萱處理 Surface Pro 保固"},
    {"query": "合約續約租金調漲", "expected_labels": ["contract_renewal"],
     "description": "和碩鍾佩珊確認續約條件"},
    {"query": "訂單數量增加300片", "expected_labels": ["order_change"],
     "description": "華碩江柏翰調整訂單數量"},
    {"query": "年度業務檢討會議", "expected_labels": ["annual_review"],
     "description": "聯發吳佳穎預約年度檢討"},
    {"query": "詐騙連結", "expected_labels": ["old_friend_reconnect"],
     "description": "高健銘假冒老同學發送釣魚連結"},
    {"query": "投資方案高回報", "expected_labels": ["investment_opportunity"],
     "description": "羅思婷推銷高回報投資方案"},
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
            str(meta.get("customer_name", "") or meta.get("external_userid", ""))[:12],
            str(meta.get("label", "-")),
            text_preview,
            graph_info,
        ])

    return tabulate(
        rows,
        headers=["#", "ID", "Score", "From", "Label", "Content", "Info"],
        tablefmt="simple_grid",
        maxcolwidths=[None, 18, None, 10, 12, 55, 25],
    )


TEST_QUERIES = [
    {"query": "诈骗", "expected_ids": [], "description": "scam-related"},
    {"query": "合作方案", "expected_ids": [], "description": "cooperation plan"},
    {"query": "product inquiry", "expected_ids": [], "description": "product_inquiry label"},
    {"query": "出货", "expected_ids": [], "description": "shipping logistics"},
    {"query": "投诉", "expected_ids": [], "description": "complaints"},
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
@click.option("--label", default=None)
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
            found_labels = set()
            for r in results:
                lbl = r.get("metadata", {}).get("label", "")
                if lbl:
                    found_labels.add(lbl)
            matched = any(el in found_labels for el in test["expected_labels"])
            if matched:
                hit_count += 1
            rows.append([
                "✅" if matched else "❌",
                test["query"][:25],
                test["description"][:30],
                ", ".join(test["expected_labels"]),
                ", ".join(sorted(found_labels)[:4]) or "-",
            ])
        click.echo(tabulate(rows, headers=["", "Query", "Description", "Expected Labels", "Found Labels"], tablefmt="simple_grid"))
        total = len(SYNTHETIC_TEST_QUERIES)
        click.echo(f"\n📊 召回率: {hit_count}/{total} = {hit_count/total*100:.1f}%")
    except Exception as e:
        logger.exception("合成基准失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()