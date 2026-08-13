# ──────────────────── Configuration ────────────────────
"""Live runtime config: MySQL connection + HF cache auto-detection.

The OCR/invoice-era constants that used to live here belonged to the
pre-migration codebase and had no remaining importers — removed.
"""
import os

import pymysql


# ── MySQL — contract screening data source ──

def build_db_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "oa_rag"),
        "charset": os.getenv("DB_CHARSET", "utf8mb4"),
        "cursorclass": pymysql.cursors.DictCursor,
    }


# ── HuggingFace offline auto-detection ──

def ensure_hf_offline() -> None:
    """Set HF_HUB_OFFLINE from cache state when it is not explicitly set.

    A fresh container starts with an empty hf-cache volume: forcing offline
    there breaks the first model load; forcing online on a primed cache adds
    hub latency to every start. Detect instead. Must run before any
    transformers / sentence-transformers / txtai import.
    """
    if os.getenv("HF_HUB_OFFLINE"):
        return
    try:
        cache = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        slug = "models--" + os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").replace("/", "--")
        present = os.path.isdir(os.path.join(cache, "hub", slug))
        os.environ["HF_HUB_OFFLINE"] = "1" if present else "0"
    except Exception:
        pass
