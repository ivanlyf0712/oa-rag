# ──────────────────── Embedding Module ────────────────────
import requests
import psycopg2

from core.config import OLLAMA_URL, EMBED_MODEL, DB_CONFIG


def get_embedding(text: str) -> list:
    """Generate embedding via Ollama mxbai-embed-large (1024-dim)."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL, "input": text
    })
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def update_embedding(row_id: int):
    """Fetch raw_text for a row, generate its embedding, and update the row."""
    conn = psycopg2.connect(**DB_CONFIG)
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