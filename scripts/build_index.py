#!/usr/bin/env python3
"""Build/rebuild the search index from MySQL.

Local:  venv/bin/python scripts/build_index.py --force [--graph-mode off]
Docker: make index   (runs this inside the app container)
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from core.config import ensure_hf_offline

ensure_hf_offline()


def main() -> int:
    from apps.search import DEFAULT_INDEX_PATH, IndexBuilder

    force = "--force" in sys.argv
    graph_mode = "off" if "--graph-mode" in sys.argv and "off" in sys.argv else "auto"
    index_path = os.getenv("INDEX_PATH") or DEFAULT_INDEX_PATH
    print("Building index at %s (force=%s, graph=%s) ..." % (index_path, force, graph_mode), flush=True)
    embeddings = IndexBuilder(index_path=index_path).build(
        force=force, enable_graph=graph_mode != "off", graph_mode=graph_mode)
    print("Index build complete: %d chunks" % embeddings.count(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
