#!/usr/bin/env python3
"""Build the OA contract risk screening txtai search index."""

import argparse
import importlib.util
import logging
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("build-index")


def _build_via_search_module(force: bool = False,
                              graph_mode: str = "auto",
                              chunk_size: int = 256,
                              index_path: str | None = None) -> int:
    search_path = os.path.join(ROOT_DIR, "apps", "search.py")
    spec = importlib.util.spec_from_file_location("search_module", search_path)
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


def main():
    parser = argparse.ArgumentParser(
        description="构建 OA 合同风险筛查搜索索引 (增强分块模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 apps/build_index.py
  python3 apps/build_index.py --force
  python3 apps/build_index.py --chunk-size 512
  python3 apps/build_index.py --graph-mode llm
  python3 apps/build_index.py --graph-mode off
  python3 apps/search.py build --force
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
        "--index-path", default=None,
        help="索引保存路径 (默认: 自动选择)"
    )

    args = parser.parse_args()

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
        print("\n现在可以运行搜索:")
        print("   python3 apps/search.py search \"查询词\" --mode hybrid --expand --rerank")
    except ImportError as e:
        logger.error(f"依赖缺失: {e}")
        logger.error("请安装依赖: pip install txtai pymysql click tabulate chonkie sentence-transformers jieba")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"索引构建失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
