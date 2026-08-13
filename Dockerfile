# OA Contract RAG — container image
#   docker build -t oa-rag .
#   (usually via: make up / docker compose up -d --build)
#
# Layout mirrors the corpchat-rag reference: slim python base, explicit
# dependency set, app code + index + uploads baked in (named volumes copy them
# up on first run so they persist across rebuilds), warmup then streamlit.

FROM python:3.12-slim

WORKDIR /app

# requirements.txt pins target CPython 3.14 (local venv); the image uses 3.12,
# so install the same key packages with pip resolving compatible versions
# (same approach as corpchat-rag). All deps have manylinux wheels — no gcc.
RUN pip install --no-cache-dir     "txtai[graph]==9.12.0" "sentence-transformers==5.6.1" "chonkie==1.7.0"     "jieba==0.42.1" "streamlit==1.59.2" "pymysql==1.1.1" "cryptography"     "pandas" "numpy" "networkx==3.6.1" "requests" "httpx" "tabulate" "tenacity"     "click" "python-dotenv" "pymupdf" "python-docx" "pillow"     "langchain-core==1.5.3" "langchain-ollama==1.1.0" "langgraph==1.2.11" "ollama==0.6.2"     "GitPython" "Faker"

# App code + built index + uploaded contract files.
COPY apps/ apps/
COPY core/ core/
COPY scripts/ scripts/
COPY uploads/ uploads/

# INDEX_PATH: in-container location (host .env points at a host path).
# HF_HUB_OFFLINE deliberately unset: core.config.ensure_hf_offline() detects
# the hf-cache volume state at startup (offline once bge-m3 is cached).
ENV STREAMLIT_SERVER_PORT=8501     STREAMLIT_SERVER_ADDRESS=0.0.0.0     PYTHONUNBUFFERED=1     PYTHONPATH=/app     INDEX_PATH=/app/apps/search_index

EXPOSE 8501

# Warmup: fail-fast on broken imports/missing index, preheat models + index
# (model-download failure only warns — the app retries lazily with a spinner).
# --server.fileWatcherType=none: avoid Streamlit scanning site-packages, which
# triggers lazy imports of all transformers submodules (~45s stall).
CMD ["sh", "-c", "python apps/warmup.py && exec streamlit run apps/app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType=none"]
