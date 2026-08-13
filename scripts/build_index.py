#!/usr/bin/env python3
"""Build/rebuild the search index from MySQL.

Local:  venv/bin/python scripts/build_index.py --force
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
    from apps.search import DEFAULT_INDEX_PATH, Searcher

    force = "--force" in sys.argv
    index_path = os.getenv("INDEX_PATH") or DEFAULT_INDEX_PATH
    print("Building index at %s (force=%s) ..." % (index_path, force))
    Searcher(index_path).build(force=force)
    print("Index build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
