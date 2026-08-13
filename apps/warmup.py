#!/usr/bin/env python3
"""Container warmup: fail-fast on broken image/config, preheat models + index.

Runs before the streamlit process (see Dockerfile CMD). Exit codes:
  1 — imports broken or index directory missing: the container should not start
  0 — ready; a model download/load failure only warns (the app retries lazily
      on first search with a Streamlit spinner, and hf-cache persists the result)
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))  # no-op in docker (env comes from compose)
except ImportError:
    pass

from core.config import ensure_hf_offline

ensure_hf_offline()  # decide HF_HUB_OFFLINE from the hf-cache state when unset


def main() -> int:
    t0 = time.time()
    try:
        from apps.search import DEFAULT_INDEX_PATH, load_index
    except Exception as exc:
        print("warmup: FATAL import error: %s" % exc)
        return 1

    index_path = os.getenv("INDEX_PATH") or DEFAULT_INDEX_PATH
    if not os.path.isdir(index_path):
        print("warmup: FATAL index directory missing: %s" % index_path)
        return 1

    try:
        t1 = time.time()
        load_index(index_path)
        print("warmup: index + embedding model loaded in %.1fs" % (time.time() - t1))
    except Exception as exc:
        print("warmup: WARN index/model load failed (%s) — app will retry lazily" % exc)

    print("warmup: done in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
