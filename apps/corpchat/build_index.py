#!/usr/bin/env python3
"""
build_index.py — 构建 txtai 搜索索引 (兼容 search.py 的增强索引)
=================================================================
基于 search.py 的 IndexBuilder 类，提供句子级分块和丰富化功能。

用法:
  # 直接运行 (独立脚本)
  python3 apps/corpchat/build_index.py [--force] [--chunk-size 256] [--graph-mode auto]

  # 或者通过 search.py 的 build 子命令 (效果相同)
  python3 apps/corpchat/search.py build [--force] [--chunk-size 256] [--graph-mode auto]

前置条件:
  pip install txtai psycopg2 click tabulate chonkie sentence-transformers
  PostgreSQL 数据库须运行且可通过 DB_CONFIG 访问 (在 core.config 或环境变量中设置)。

依赖关系:
  - 此脚本产生的索引完全兼容 search.py 的搜索功能 (包括多查询扩展、RRF 融合、图增强、重排序)
  - 与旧版 search.py (未增强版) 的索引格式不兼容，因为增加了分块和丰富化
"""

import os
import sys
import json
import logging

# ── 确保项目根目录在 Python 路径上 ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.config import DB_CONFIG

# ── 日志 ──
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("build-index")


# ═══════════════════════════════════════════════════════════════════
# 复用 search.py 中的 IndexBuilder
# 参考 analysis_report.md §2.2 (分块) 和 §2.3 (丰富化)
# ═══════════════════════════════════════════════════════════════════

def _build_via_search_module(force: bool = False,
                              graph_mode: str = "auto",
                              chunk_size: int = 256,
                              index_path: str | None = None) -> int:
    """
    通过 search.py 的 IndexBuilder 构建索引。

    这种方式会执行句子级分块 + 丰富化，与 search.py 的搜索功能完全兼容。

    Returns: 索引中的文档块总数
    """
    # 动态导入以避免在 import 阶段就产生依赖
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    # 从 search.py 导入 IndexBuilder (位于同一目录)
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search_module",
        os.path.join(os.path.dirname(__file__), "search.py")
    )
    search_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search_module)

    builder = search_module.IndexBuilder(
        index_path=index_path or search_module.DEFAULT_INDEX_PATH,
        chunk_size=chunk_size,
    )
    enable_graph = graph_mode != "off"
    embeddings = builder.build(force=force, enable_graph=enable_graph, graph_mode=graph_mode)
    count = embeddings.count()
    logger.info(f"索引构建完成: {count} 个文档块 (图: {'✅' if embeddings.graph else '❌'})")
    return count


# ═══════════════════════════════════════════════════════════════════
# 向后兼容: 旧的直接构建方式 (无分块/丰富化)
# 保留此方法以防止已有的索引依赖
# ═══════════════════════════════════════════════════════════════════

def _build_legacy(dsn: str, model_path: str, index_dir: str) -> int:
    """
    旧版索引构建: 直接索引原始消息文本，不分块不丰富化。

    此模式已弃用，仅用于兼容旧版 search.py (未增强版) 的使用。
    推荐使用 _build_via_search_module() 替代。
    """
    import txtai
    import psycopg2

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 读取联系人
    cur.execute("SELECT full_name, job_title, company, phone, email, userid FROM contacts")
    contacts = [dict(zip([col[0] for col in cur.description], row)) for row in cur.fetchall()]

    # 读取消息
    cur.execute("SELECT content, label, external_userid, servicer_userid, send_time FROM messages")
    messages = [dict(zip([col[0] for col in cur.description], row)) for row in cur.fetchall()]

    cur.close()
    conn.close()

    if not contacts and not messages:
        logger.warning("数据库中无数据")
        return 0

    embeddings = txtai.Embeddings({
        "path": model_path,
        "content": dsn,
        "writable": True,
        "hybrid": True,
        "scoring": {"method": "bm25"},
    })

    # 联系人文档
    contact_docs = [
        (
            f"contact_{i}",
            f"{c['full_name']} {c['job_title']} {c['company']} "
            f"{c['phone']} {c['email']} (userid: {c['userid']})",
            json.dumps(c, default=str),
        )
        for i, c in enumerate(contacts)
    ]

    logger.info(f"索引 {len(contact_docs)} 个联系人 ...")
    if contact_docs:
        embeddings.index(contact_docs)

    # 消息文档 (upsert)
    message_docs = [
        (
            f"message_{i}",
            f"[{m.get('label', '')}] {m['external_userid']} ↔ "
            f"{m.get('servicer_userid', 'system')}: {m['content']}",
            json.dumps(m, default=str),
        )
        for i, m in enumerate(messages)
    ]

    if message_docs:
        logger.info(f"索引 {len(message_docs)} 条消息 (upsert)...")
        embeddings.upsert(message_docs)

    embeddings.save(index_dir)
    count = embeddings.count()
    logger.info(f"旧版索引构建完成 (共 {count} 条文档): {index_dir}")
    return count


# ═══════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="构建 CorpChat 搜索索引 (支持增强分块模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 apps/corpchat/build_index.py                                  # 增强模式 (默认)
  python3 apps/corpchat/build_index.py --force                          # 强制重建
  python3 apps/corpchat/build_index.py --chunk-size 512                 # 自定义分块大小
  python3 apps/corpchat/build_index.py --graph-mode llm                 # LLM 图关系提取
  python3 apps/corpchat/build_index.py --graph-mode off                 # 禁用图
  python3 apps/corpchat/build_index.py --legacy                         # 旧版模式 (不分块)
  python3 apps/corpchat/search.py build --force                         # 等效命令 (search.py)
        """,
    )
    parser.add_argument("--force", action="store_true", help="强制重建索引")
    parser.add_argument(
        "--graph-mode", choices=["auto", "llm", "off"], default="auto",
        help="图模式: auto=向量推断, llm=LLM 提取关系, off=禁用 (默认: auto)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help="分块 token 数 (默认: 256)"
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="使用旧版模式 (不分块/丰富化, 纯 txtai 索引)"
    )
    parser.add_argument(
        "--index-path", default=None,
        help="索引保存路径 (默认: 自动选择)"
    )

    args = parser.parse_args()

    # ── 旧版兼容模式 ──
    if args.legacy:
        logger.info("使用旧版模式 (不分块, 不丰富化)")
        dsn = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        )
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
        local_model = os.path.join(ROOT_DIR, "models", "all-MiniLM-L6-v2")
        if os.path.isdir(local_model):
            model_path = local_model
        index_dir = args.index_path or os.path.join(os.path.dirname(__file__), "corpchat_index")
        count = _build_legacy(dsn, model_path, index_dir)
        print(f"\n✅ 完成 (旧版模式): {count} 条文档")
        print(f"⚠️  注意: 旧版索引与 search.py 的增强搜索功能不兼容")
        print(f"   建议使用 search.py: python3 apps/corpchat/search.py build --force")
        return

    # ── 增强模式 (默认) ──
    try:
        count = _build_via_search_module(
            force=args.force,
            graph_mode=args.graph_mode,
            chunk_size=args.chunk_size,
            index_path=args.index_path,
        )
        print(f"\n✅ 增强索引构建完成: {count} 个文档块")
        print(f"   索引路径: {args.index_path or '(默认搜索路径)'}")
        print(f"   图模式: {args.graph_mode}")
        print(f"   分块大小: {args.chunk_size} tokens")
        print(f"\n现在可以运行搜索:")
        print(f"   python3 apps/corpchat/search.py search \"查询词\" --mode auto --expand --rerank")

    except ImportError as e:
        logger.error(f"依赖缺失: {e}")
        logger.error("请安装依赖: pip install txtai click tabulate chonkie sentence-transformers")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"索引构建失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()