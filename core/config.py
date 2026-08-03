# ──────────────────── Configuration ────────────────────
import os

# ── OCR: Server mode (primary, faster) ──
OCR_MODE = "server"               # "server" | "cli"
OCR_SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"
OCR_SERVER_MODEL = "Unlimited-OCR"
OCR_SERVER_PROMPT = "Please OCR the text in this image."
OCR_SERVER_TEMPERATURE = 0.0
OCR_SERVER_MAX_TOKENS = 32768
OCR_SERVER_REPEAT_PENALTY = 1.1

# ── OCR: CLI mode (fallback, uses subprocess) ──
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
UOCR_MODEL = os.path.expanduser("~/uocr/Unlimited-OCR-Q4_K_M.gguf")
UOCR_MMPROJ = os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf")

# ── Image preprocessing ──
MAX_LONG_EDGE = 512              # server can handle larger images
JPEG_QUALITY = 60

# Ollama
OLLAMA_URL = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen2.5:1.5b"      # JSON extraction model
EMBED_MODEL = "mxbai-embed-large"
RAG_MODEL = "qwen2.5:1.5b"       # RAG answer generation model

# llama-server for multi-page mode (used by pipeline_fast.py)
LLAMA_SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"

# PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ocr",
    "password": "ocrpass",
    "dbname": "invoices"
}

# JSON extraction prompts
JSON_PROMPT = """Return a single JSON object with these keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".

Rules:
- Use the exact text from the invoice. Do NOT invent or guess any values.
- If a field is missing, set it to "".
- "total_amount" must contain only the number (e.g. "1250.00"), without currency symbol.
- "currency" must be the three‑letter currency code (e.g. "USD").
- Do NOT use nested objects.

Invoice text:
___RAW_TEXT___

JSON:"""

FALLBACK_PROMPT = """Extract these fields from the invoice text.
Do NOT use any of the following words: value, text, string, example, placeholder, xxxx.
Return ONLY a valid JSON object with the keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".
"total_amount" must be a plain number (e.g. "1250.00").
"currency" must be a three‑letter code (e.g. "USD").
If a field is truly missing, leave it as "".

Invoice text:
___RAW_TEXT___

JSON:"""