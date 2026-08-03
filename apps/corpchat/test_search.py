#!/usr/bin/env python3
"""
Validation script for txtai search index (PostgreSQL-backed).

Tests both BM25 (keyword) and semantic (vector) search modes against the
corpchat index. Prints results for each query and a summary.

Usage:
    cd /home/ivanleeyf/ocr
    source venv/bin/activate
    python3 apps/corpchat/test_search.py
"""

import sys
import os
import json
import txtai

# ── Ensure the project root (ocr/) is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.config import DB_CONFIG


def search_and_print(embeddings, label, query, limit=5, weights=None):
    """
    Run a search with optional hybrid weights and print results.

    weights=None       → balanced hybrid (default)
    weights=(0.0, 1.0) → pure BM25  (keyword/sparse only)
    weights=(1.0, 0.0) → pure vector (dense/ANN only)
    """
    print(f"\n{'=' * 65}")
    print(f"  [{label}] query = \"{query}\"", end="")
    if weights is not None:
        print(f"  |  weights={weights}", end="")
    print()
    print(f"{'=' * 65}")

    results = embeddings.search(query, limit=limit, weights=weights)

    if not results:
        print("  ⚠️  No results returned.")
        return results

    for rank, r in enumerate(results, start=1):
        doc_id = r.get("id", "?")
        text = r.get("text", "") or ""
        score = r.get("score", 0.0)

        # Tags (metadata JSON) stored as a string in the database
        tags_raw = r.get("tags", "")
        meta = ""
        if tags_raw:
            try:
                meta_dict = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                # Show a concise preview of metadata
                meta_parts = []
                if "full_name" in meta_dict:
                    meta_parts.append(f"name={meta_dict['full_name']}")
                if "userid" in meta_dict:
                    meta_parts.append(f"userid={meta_dict['userid']}")
                if "company" in meta_dict:
                    meta_parts.append(f"company={meta_dict['company']}")
                if "label" in meta_dict:
                    meta_parts.append(f"label={meta_dict['label']}")
                if "external_userid" in meta_dict:
                    meta_parts.append(f"from={meta_dict['external_userid']}")
                if "content" in meta_dict:
                    meta_parts.append(f"content={meta_dict['content'][:60]}")
                meta = " | ".join(meta_parts)
            except (json.JSONDecodeError, TypeError):
                meta = str(tags_raw)[:80]

        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  [{rank}] id={doc_id}  score={score:.4f}")
        print(f"        text: {text_preview}")
        if meta:
            print(f"        meta: {meta}")

    return results


def main():
    # ── Build PostgreSQL DSN from DB_CONFIG ──
    dsn = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )

    print("=" * 65)
    print("  CorpChat Search Index — Validation Suite")
    print("=" * 65)
    print(f"  PostgreSQL: {dsn.split('@')[0].split(':')[0]}:***@{dsn.split('@')[1]}")
    print()

    # ── Load the embeddings index ──
    # When loading, txtai will read the saved config and reconnect to PostgreSQL.
    index_path = os.path.join(os.path.dirname(__file__), "corpchat_index")
    if not os.path.isdir(index_path):
        print(f"❌ Index directory not found at {index_path}")
        print("   Run build_index.py first to create the index.")
        sys.exit(1)

    print(f"Loading index from {index_path} ...")
    embeddings = txtai.Embeddings()
    embeddings.load(index_path)
    total = embeddings.count()
    print(f"  ✅ Index loaded — {total} document(s) indexed.\n")

    # ── Define test queries ──
    # These should hit both contact records and message records

    test_queries_contacts = [
        "傅健",                     # Chinese name from contacts
        "manager",                  # English job title keyword
        "John",                     # common English first name
        "finance",                  # company name keyword
    ]

    test_queries_messages = [
        "詐騙",                     # "scam" — Chinese message keyword
        "project deadline",         # English message content keyword
        "investment opportunity",   # semantic: may match "crypto" or "scam"
        "合作",                     # "cooperation" — Chinese keyword
    ]

    all_ok = True

    # ── 1. BM25 (keyword) search ──
    print("\n" + "#" * 65)
    print("#  SECTION 1: BM25 (Keyword) Search  —  weights=(0.0, 1.0)")
    print("#" * 65)

    for query in test_queries_contacts:
        r = search_and_print(embeddings, "BM25-CONTACT", query, limit=3, weights=(0.0, 1.0))
        if not r:
            print("  ⚠️  BM25 contact search returned no results.")
            all_ok = False

    for query in test_queries_messages:
        r = search_and_print(embeddings, "BM25-MESSAGE", query, limit=3, weights=(0.0, 1.0))
        if not r:
            print("  ⚠️  BM25 message search returned no results.")
            all_ok = False

    # ── 2. Vector (semantic) search ──
    print("\n" + "#" * 65)
    print("#  SECTION 2: Vector (Semantic) Search  —  weights=(1.0, 0.0)")
    print("#" * 65)

    for query in test_queries_contacts:
        r = search_and_print(embeddings, "VECTOR-CONTACT", query, limit=3, weights=(1.0, 0.0))
        if not r:
            print("  ⚠️  Vector contact search returned no results.")
            # Not required to fail — may be expected if embeddings don't match

    for query in test_queries_messages:
        r = search_and_print(embeddings, "VECTOR-MESSAGE", query, limit=3, weights=(1.0, 0.0))
        if not r:
            print("  ⚠️  Vector message search returned no results.")

    # ── 3. Hybrid (default) search ──
    print("\n" + "#" * 65)
    print("#  SECTION 3: Hybrid (Default) Search  —  weights=None")
    print("#" * 65)

    hybrid_queries = [
        ("Zhang",                  "contact name keyword"),
        ("report",                 "message content keyword"),
        ("urgent",                 "message content keyword"),
        ("meeting",                "message content keyword"),
    ]
    for query, desc in hybrid_queries:
        r = search_and_print(embeddings, f"HYBRID ({desc})", query, limit=3)
        if not r:
            print(f"  ⚠️  Hybrid search for \"{query}\" returned no results.")

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total documents in index: {total}")
    print(f"  BM25 search:    {'✅ working' if all_ok else '⚠️  issues detected'}")
    print(f"  Vector search:  check individual results above")
    print(f"  Hybrid search:  check individual results above")
    print(f"\n  If BM25 returns results but vector does not, the hybrid config may")
    print(f"  need the 'scoring' component enabled (it should be with hybrid=True).")
    print(f"  If both return results, the index is fully operational.")


if __name__ == "__main__":
    main()