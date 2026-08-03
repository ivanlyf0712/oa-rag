# ──────────────────── Database Module ────────────────────
import psycopg2
import pandas as pd
import requests
import warnings

from core.config import DB_CONFIG, OLLAMA_URL, EMBED_MODEL

# Suppress pandas+psycopg2 warning (pd.read_sql works fine with raw connections)
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def insert_invoice(fields: dict, raw_text: str, source_file: str):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO invoices (invoice_number, date, vendor_name,
                                  total_amount, currency, raw_text, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            fields.get("invoice_number", ""),
            fields.get("date", ""),
            fields.get("vendor_name", ""),
            fields.get("total_amount", ""),
            fields.get("currency", ""),
            raw_text,
            source_file
        ))
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def fetch_all_invoices():
    conn = get_db_connection()
    df = pd.read_sql("SELECT id, invoice_number, date, vendor_name, total_amount, currency, source_file, created_at FROM invoices ORDER BY created_at DESC", conn)
    conn.close()
    return df


def get_embedding(text: str):
    """Generate embedding via Ollama mxbai-embed-large (1024-dim)."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL, "input": text
    })
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def update_embedding(row_id: int):
    """Fetch raw_text for a row, generate its embedding, and update the row."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT raw_text FROM invoices
        WHERE id = %s AND embedding IS NULL
    """, (row_id,))
    row = cur.fetchone()
    if row and row[0]:
        raw_text = row[0].strip()
        if raw_text:
            # Embed the full OCR text for richer semantic search
            # Truncate to ~4K chars to stay within embedding model's context window
            text_to_embed = raw_text[:4096]
            vec = get_embedding(text_to_embed)
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))
            conn.commit()
    cur.close()
    conn.close()


def search_similar(query, vendor_filter=None, top_k=5,
                   date_from=None, date_to=None, amount_min=None, amount_max=None,
                   keyword_filter=None):
    """
    Hybrid semantic + keyword search over invoices using pgvector cosine similarity
    and PostgreSQL full‑text search on raw_text.

    Args:
        query: Natural language search query (embedded via mxbai-embed-large).
        vendor_filter: Optional ILIKE pattern for vendor name.
        top_k: Number of results to return (1-100).
        date_from: Optional start date filter (YYYY-MM-DD).
        date_to: Optional end date filter (YYYY-MM-DD).
        amount_min: Optional minimum total_amount filter.
        amount_max: Optional maximum total_amount filter.
        keyword_filter: Optional keyword/phrase for full‑text search on raw_text.
                        When provided, results must match BOTH the semantic query
                        AND contain the keyword in the OCR text.

    Returns:
        List of tuples: (id, invoice_number, date, vendor_name, total_amount, currency, similarity)
    """
    query_vec = get_embedding(query)
    conn = get_db_connection()
    cur = conn.cursor()

    # Build WHERE clauses dynamically for structured filters
    conditions = ["embedding IS NOT NULL"]
    params: list = [query_vec]

    if vendor_filter:
        conditions.append("vendor_name ILIKE %s")
        params.append(f"%{vendor_filter}%")

    if date_from:
        conditions.append("date >= %s")
        params.append(date_from)

    if date_to:
        conditions.append("date <= %s")
        params.append(date_to)

    if amount_min:
        conditions.append("total_amount::numeric >= %s")
        params.append(amount_min)

    if amount_max:
        conditions.append("total_amount::numeric <= %s")
        params.append(amount_max)

    if keyword_filter:
        # Use PostgreSQL full‑text search with plainto_tsquery for user‑friendly input.
        # to_tsvector('english') on raw_text creates a text‑search vector;
        # plainto_tsquery converts the keyword into a tsquery (AND‑ing all terms).
        conditions.append(
            "to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"
        )
        params.append(keyword_filter)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT id, invoice_number, date, vendor_name, total_amount, currency,
               1 - (embedding <=> %s::vector) AS similarity
        FROM invoices
        WHERE {where_clause}
        ORDER BY similarity DESC
        LIMIT %s
    """
    params.append(top_k)

    cur.execute(sql, params)
    results = cur.fetchall()
    cur.close()
    conn.close()

    return results if results else []