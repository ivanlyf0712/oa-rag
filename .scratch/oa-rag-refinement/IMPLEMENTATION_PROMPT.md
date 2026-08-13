# Agent Instructions: Migrate corpchat-rag → oa-rag (Contract Risk Screening)

> **IMPORTANT BEFORE YOU START**
> Before writing ANY code, **read the following files in order**. They contain the full context of what you're modifying:
> 1. `/corpchat-rag/apps/corpchat/search.py` (the entire 1292-line file — this is your primary source)
> 2. `/corpchat-rag/apps/corpchat/app.py` (the entire 568-line Streamlit UI)
> 3. `/corpchat-rag/core/config.py` (DB_CONFIG — PostgreSQL connection)
> 4. `/corpchat-rag/core/db.py` (PostgreSQL helper functions — partially reusable)
> 5. `/corpchat-rag/apps/corpchat/build_index.py` (index builder CLI wrapper)
> 6. `/corpchat-rag/apps/corpchat/init_chat.sql` (original DB schema pattern)
> 7. `/corpchat-rag/.env` (environment variable conventions)
> 8. `/corpchat-rag/requirements.txt` (pinned dependencies)
>
> **All source files live at `/corpchat-rag/` — NOT `/Users/ivanlee/Desktop/corpchat-rag/`.**
> Your working directory is `/Users/ivanlee/Desktop/oa-rag/`.
> The corpchat-rag source files are accessible to you — **read them, don't guess**.

---

## 1. Mission

Transform the **corpchat-rag** system (designed for WeChat Work message search) into **oa-rag** (Organizational Archives — Contract Risk Screening). The core search infrastructure (txtai embeddings, hybrid search, RRF fusion, query expansion, reranking) stays. The **data domain** changes from "chat messages" to "contracts," and the **UI** changes from chat analytics to a contract risk filtering dashboard.

### Two HARD requirements:
1. **Onyx Chat must be completely removed.** The entire "Onyx Chat" tab/page, the iframe embed, the deprecated chat functionality, and all references to `OLLAMA_URL`/`Onyx` in the UI must be gone. No trace.
2. **MySQL contracts must be fully integrated into the search pipeline.** The search must read contracts from a MySQL `contracts` table, index them with txtai, and allow metadata filtering (risk_level, amount, audit_status, legal_approval, overruled, department, tags).

### Your environment:
- You have access to the **real MySQL `contracts` table**. Run `DESCRIBE contracts;` to inspect the exact column names, types, and values. **Do not assume** — verify every column type, every ENUM value, every nullable field.
- The `corpchat-rag` source files are available for reading at `/corpchat-rag/`.

---

## 2. Target Architecture

```
oa-rag/
├── search.py                    # [MODIFY] Core search engine — PostgreSQL→MySQL, messages→contracts
├── .env.example                 # [CREATE] MySQL connection + search config (based on corpchat-rag/.env)
├── requirements.txt             # [MODIFY] Replace psycopg2-binary → pymysql
├── PROJECT_OVERVIEW.md          # [CREATE] Project documentation
├── core/
│   ├── __init__.py              # [COPY] Already done
│   ├── config.py                # [MODIFY] DB_CONFIG → MySQL (port 3306, pymysql)
│   ├── db.py                    # [MODIFY] Replace search_similar() with contract helpers
│   └── embedding.py             # [COPY] Keep as-is (Ollama embedding helper)
├── apps/oa/
│   ├── __init__.py              # [COPY] Already done
│   ├── search.py                # [COPY + MODIFY] Copy corpchat search.py, then adapt
│   ├── app.py                   # [REWRITE] New Streamlit UI — contracts only, NO Onyx Chat
│   ├── build_index.py           # [MODIFY] Update path references & DB driver
│   └── gen_fake_contract.py     # [CREATE] MySQL fake contract data generator
└── .scratch/oa-rag-refinement/
    └── IMPLEMENTATION_PROMPT.md # [THIS FILE]
```

---

## 3. Step-by-Step Implementation Guide

### Step 1: Create `core/config.py` (MySQL configuration)

**Base file:** `/corpchat-rag/core/config.py`

**Changes required:**
- Change `DB_CONFIG` to use MySQL: port `3306`, add `"unix_socket": None` (or `charset`), remove PostgreSQL-specific settings.
- Replace all PostgreSQL-specific references. Keep OCR/LLM config from the original (they're reusable).
- The original `DB_CONFIG` (lines 31-38 of corpchat-rag/core/config.py) looks like:

```python
# PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ocr",
    "password": "ocrpass",
    "dbname": "invoices"
}
```

You must produce:

```python
# MySQL — CONTRACT RISK SCREENING
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "contracts_db"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,  # OPTIONAL — use if you want dict rows
}
```

**Key difference:** MySQL uses `database` (not `dbname`), and the pymysql driver uses `charset` instead of `client_encoding`. You must import `pymysql` at the top of the file. The driver name is `pymysql` (imported as `import pymysql`).

For the `core/db.py` connection function, change from:
```python
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)
```
to:
```python
def get_db_connection():
    import pymysql
    return pymysql.connect(**DB_CONFIG)
```

**Action items:**
- [ ] Set `DB_PORT` default to `3306` (MySQL default)
- [ ] Set `DB_NAME` default to whatever your MySQL database is called (verify via `SHOW DATABASES;`)
- [ ] Keep `OLLAMA_URL`, `TEXT_MODEL`, `EMBED_MODEL`, `RAG_MODEL`, OCR config — these are reused as-is
- [ ] You may want to add a `CONTRACTS_TABLE = os.getenv("CONTRACTS_TABLE", "contracts")` constant

---

### Step 2: Create `.env.example`

**Base file:** `/corpchat-rag/.env` (this is the actual .env, not a .example)

Create `/Users/ivanlee/Desktop/oa-rag/.env.example` based on the corpchat-rag `.env`, but **swap PostgreSQL for MySQL variables**:

```bash
# MySQL — Contract database
DB_HOST=localhost
DB_PORT=3306
DB_NAME=contracts_db
DB_USER=root
DB_PASSWORD=
DB_CHARSET=utf8mb4

# LiteLLM / LLM API config (used for query expansion & reranking)
LITELLM_API_KEY=
LITELLM_BASE_URL=http://localhost:11434
LITELLM_MODEL=qwen2.5:1.5b

# Embedding model (txtai local model or HuggingFace ID)
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-base
INDEX_PATH=apps/oa/search_index

# Contract table name (optional override)
CONTRACTS_TABLE=contracts
```

**Important:** The corpchat-rag `.env` currently has `LITELLM_API_KEY=ollama` and `LITELLM_BASE_URL=http://localhost:11434` with `LITELLM_MODEL=qwen2.5:1.5b` — keep those same values in oa-rag since the LLM setup is the same.

---

### Step 3: Create `requirements.txt`

**Base file:** `/corpchat-rag/requirements.txt`

**Changes required:**
- Replace `psycopg2-binary==2.9.12` with `pymysql==1.1.1` (or latest compatible)
- Keep everything else (txtai, click, tabulate, chonkie, jieba, sentence-transformers, streamlit, python-dotenv, pandas, etc.)
- `chonkie` — needed for sentence-level chunking
- `jieba` — needed for Chinese segmentation (keep even if contracts are English — it doesn't hurt, and the original uses it)
- `sentence-transformers` — needed for the cross-encoder reranker

**Action items:**
- [ ] Remove `psycopg2-binary==2.9.12` (line 36 of original)
- [ ] Add `pymysql==1.1.1`
- [ ] Keep `txtai==9.12.0`, `click==8.4.2`, `tabulate==0.10.0`, `chonkie==1.7.0`, `jieba==0.42.1`, `sentence-transformers==5.6.1`, `streamlit==1.59.2`, `python-dotenv==1.2.2`, `pandas==3.0.3`
- [ ] Install deps: `pip install -r requirements.txt`

---

### Step 4: Create `core/db.py` (MySQL contract helpers)

**Base file:** `/corpchat-rag/core/db.py`

The original `db.py` has PostgreSQL-specific functions for the `invoices` table. For oa-rag, you need to **replace** the PostgreSQL-specific helpers with MySQL-based contract helpers. The `get_embedding()` and `update_embedding()` functions at the bottom can be **removed or adapted** — they reference `invoices` table which doesn't exist in the contract domain.

**Action items:**
- [ ] Change `import psycopg2` → `import pymysql`
- [ ] Change `get_db_connection()` to use `pymysql.connect(**DB_CONFIG)` instead of `psycopg2.connect(**DB_CONFIG)`
- [ ] Remove `insert_invoice()` (not relevant to contracts)
- [ ] Remove `fetch_all_invoices()` (not relevant)
- [ ] Remove `get_embedding()` (uses Ollama — this is handled by txtai in search.py)
- [ ] Remove `update_embedding()` (not relevant)
- [ ] Remove `search_similar()` (this is PostgreSQL/pgvector-specific; oa-rag uses txtai, not pgvector)
- [ ] **Add** a `fetch_contracts(filters: dict = None) -> List[Dict]` function that queries the MySQL `contracts` table

For `fetch_contracts`, you need to:
1. Run `DESCRIBE contracts;` to get the **real** schema
2. Write a query like:
```python
def fetch_contracts(limit: int = 10000) -> List[Dict]:
    """Fetch all contracts from MySQL for indexing."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # IMPORTANT: replace column names with the ACTUAL columns from DESCRIBE contracts
    cursor.execute("""
        SELECT id, title, content, amount, risk_level, risk_warning,
               audit_status, legal_approval, overruled, department,
               sign_date, tags
        FROM contracts
        WHERE content IS NOT NULL AND content != ''
        LIMIT %s
    """, (limit,))
    # With pymysql cursors (non-DictCursor), fetch by index:
    rows = cursor.fetchall()
    # Determine column names from cursor.description
    columns = [desc[0] for desc in cursor.description]
    contracts = [dict(zip(columns, row)) for row in rows]
    cursor.close()
    conn.close()
    return contracts
```

**CRITICAL:** The exact column names and types must be verified against your real MySQL table. If `legal_approval` is `TINYINT(1)`, you'll get `0`/`1` and need to convert to `True`/`False`. If `tags` is JSON, you'll get a string and need `json.loads()`. If `amount` is `DECIMAL`, pymysql returns it as a `decimal.Decimal` — convert with `float()`.

---

### Step 5: Create `init_contracts.sql`

**Base file:** `/corpchat-rag/apps/corpchat/init_chat.sql` (schema definition pattern)

Create `/Users/ivanlee/Desktop/oa-rag/apps/oa/init_contracts.sql` — a MySQL schema definition for the contracts table.

**You MUST first run `DESCRIBE contracts;` on your real MySQL database** to get the exact schema. Then create this init SQL. If you don't know the exact DDL, create a reasonable schema matching the task's field list:

```sql
-- init_contracts.sql — Schema for OA-RAG Contract Risk Screening
-- Run with: mysql -u root -p contracts_db -f init_contracts.sql

USE contracts_db;

CREATE TABLE IF NOT EXISTS contracts (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    title           VARCHAR(500) NOT NULL COMMENT 'Contract title / subject',
    content         LONGTEXT     NOT NULL COMMENT 'Full contract text content',
    amount          DECIMAL(15,2) DEFAULT NULL COMMENT 'Contract amount in local currency',
    risk_level      VARCHAR(20) DEFAULT 'low' COMMENT 'Risk severity: high/medium/low',
    risk_warning    TEXT          DEFAULT NULL COMMENT 'AI-generated risk warning text',
    audit_status    VARCHAR(50) DEFAULT 'pending' COMMENT 'Audit status: pending/audited/approved/rejected',
    legal_approval  TINYINT(1) DEFAULT 0 COMMENT '1 = approved, 0 = not approved',
    overruled       TINYINT(1) DEFAULT 0 COMMENT '1 = overruled, 0 = not overruled',
    department      VARCHAR(100) DEFAULT NULL COMMENT 'Department that owns the contract',
    sign_date       DATE          DEFAULT NULL COMMENT 'Contract signing date',
    tags            JSON          DEFAULT NULL COMMENT 'Arbitrary contract tags as JSON array',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_contracts_risk (risk_level),
    INDEX idx_contracts_audit (audit_status),
    INDEX idx_contracts_department (department),
    INDEX idx_contracts_sign_date (sign_date),
    INDEX idx_contracts_overruled (overruled),
    INDEX idx_contracts_amount (amount),
    FULLTEXT idx_contracts_title_content (title, content)
);

-- Verify with: DESCRIBE contracts;
```

**Action items:**
- [ ] Run `DESCRIBE contracts;` on your MySQL instance
- [ ] If the real schema differs from this template, update the SQL and note the differences in PROJECT_OVERVIEW.md
- [ ] If `tags` is not JSON type, adapt accordingly (e.g., comma-separated VARCHAR → store as-is, parse in Python)
- [ ] If `legal_approval` or `overruled` are VARCHAR or ENUM, handle the type conversion in Python

---

### Step 6: Copy & Adapt `apps/oa/search.py` (THE CORE FILE)

**Base file:** `/corpchat-rag/apps/corpchat/search.py` (1292 lines)

This is the most critical file. You need to **copy** it to `oa-rag/apps/oa/search.py` first, then make these specific changes. **Do not rewrite from scratch** — preserve the existing txtai indexing, hybrid search, RRF fusion, reranker, and CLI structure. Only change the data source and metadata handling.

#### 6a. Imports (lines 33-46 of original)

Change:
```python
import psycopg2
```
to:
```python
import pymysql
```

Change the logger name:
```python
logger = logging.getLogger("corpchat-search")
```
to:
```python
logger = logging.getLogger("oa-search")
```

Change `DB_CONFIG` import (lines 69-78) — the original tries `from core.config import DB_CONFIG` and falls back to a dict. **Change the fallback defaults** to MySQL:

```python
try:
    from core.config import DB_CONFIG
except ImportError:
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "database": os.getenv("DB_NAME", "contracts_db"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "charset": "utf8mb4",
    }
```

#### 6b. `IndexBuilder._fetch_messages()` → `_fetch_contracts()` (lines 221-258)

**Replace the entire method.** The original reads from a PostgreSQL `messages` table with a JOIN to `contacts`. Your version reads from MySQL `contracts` table directly — no JOIN needed.

```python
def _fetch_contracts(self) -> List[Dict]:
    """从 MySQL 读取合同数据。"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, content, amount, risk_level, risk_warning,
               audit_status, legal_approval, overruled, department,
               sign_date, tags
        FROM contracts
        WHERE content IS NOT NULL AND content != ''
    """)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    contracts = []
    for row in rows:
        raw = dict(zip(columns, row))
        sign_date_raw = raw.get("sign_date")
        if hasattr(sign_date_raw, 'isoformat'):
            sign_date_str = sign_date_raw.isoformat()
        else:
            sign_date_str = str(sign_date_raw) if sign_date_raw else None

        # Type normalization — ADAPT based on your actual column types:
        # - amount: DECIMAL → float
        # - legal_approval: TINYINT(1) → bool
        # - overruled: TINYINT(1) → bool
        # - tags: JSON string → str (keep as JSON string for txtai tags column)
        contracts.append({
            "id": raw["id"],
            "title": raw.get("title", ""),
            "content": raw.get("content", ""),
            "amount": float(raw["amount"]) if raw.get("amount") is not None else None,
            "risk_level": raw.get("risk_level"),
            "risk_warning": raw.get("risk_warning"),
            "audit_status": raw.get("audit_status"),
            "legal_approval": bool(raw["legal_approval"]) if raw.get("legal_approval") is not None else None,
            "overruled": bool(raw["overruled"]) if raw.get("overruled") is not None else None,
            "department": raw.get("department"),
            "sign_date": raw.get("sign_date"),
            "sign_date_str": sign_date_str,
            "tags": raw.get("tags"),
        })
    return contracts
```

**Critical verification points:**
- Run `DESCRIBE contracts;` and confirm each column name matches
- Check if `amount` returns `Decimal` — `float()` conversion handles it
- Check if `legal_approval`/`overruled` return `int` (0/1) — `bool()` handles it; if they're VARCHAR ('approved'/'pending'), change to `raw.get("legal_approval") == "approved"` or similar
- Check if `tags` is JSON — if it's a JSON string, keep it as a string. If it's already a Python list/dict (pymysql JSON support), serialize with `json.dumps()`

#### 6c. `IndexBuilder._chunk_message()` → `_chunk_contract()` (lines 261-326)

**Replace the entire method.** The original chunks messages by sentence using `chonkie.SentenceChunker`. For contracts, you have two options:

**Option 1 — Single chunk per contract (recommended for POC):**
```python
def _chunk_contract(self, contract: Dict) -> List[Dict]:
    """整份合同作为一个块, 或按条款分块 (如果合同较长)。"""
    content = contract["content"]

    # If contract is long (>chunk_size tokens), split by clause/section
    # The original uses token counting heuristic for Chinese:
    #   chinese_chars / 2 + other_chars / 4
    estimated_tokens = len(content) / 3  # rough English token approximation

    chunks_text = []
    if estimated_tokens > self.chunk_size * 4:
        # Long contract — try chonkie sentence chunking
        try:
            from chonkie import SentenceChunker
            def _token_counter(text: str) -> int:
                return len(text) // 4  # rough English token estimate
            chunker = SentenceChunker(
                tokenizer_or_token_counter=_token_counter,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                return_type="texts",
            )
            chunks_text = chunker.chunk(content)
        except Exception:
            # Fallback: split by double newlines (paragraph sections)
            import re as _re
            chunks_text = _re.split(r'\n\s*\n', content)
    else:
        # Short contract — single chunk
        chunks_text = [content]

    if not chunks_text:
        chunks_text = [content]

    base_id = str(contract["id"])
    contract_id = contract["id"]
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunk_id = f"contract_{base_id}__chunk{i}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "contract_id": contract_id,
                "title": contract.get("title", ""),
                "amount": contract.get("amount"),
                "risk_level": contract.get("risk_level"),
                "risk_warning": contract.get("risk_warning"),
                "audit_status": contract.get("audit_status"),
                "legal_approval": contract.get("legal_approval"),
                "overruled": contract.get("overruled"),
                "department": contract.get("department"),
                "sign_date": contract.get("sign_date_str"),
                "tags": contract.get("tags"),
                "chunk_index": i,
            },
            "title": contract.get("title", f"Contract #{contract_id}"),
        })
    return chunks
```

#### 6d. `IndexBuilder._enrich_chunk()` (lines 329-344)

**Keep this method mostly as-is.** It combines `title + content` into the enriched text. The original uses `_segment()` for Chinese jieba tokenization. Since contracts may be English or Chinese, **keep jieba** — it gracefully handles non-Chinese text (passes it through). The `_segment()` function won't harm English text.

However, update the docstring to reflect contracts instead of messages.

#### 6e. `IndexBuilder.build()` (lines 347-415)

**Changes needed:**
- Change `messages = self._fetch_messages()` → `contracts = self._fetch_contracts()`
- Change `for msg in messages: chunks = self._chunk_message(msg)` → `for contract in contracts: chunks = self._chunk_contract(contract)`
- Change logging messages: "条消息 → N 个块" → "份合同 → N 个块"
- **Remove or disable graph features:** The `_compute_structural_relationships()` function (lines 165-202) computes relationships based on `open_kfid`, `external_userid`, `servicer_userid`, `label`, `company` — none of which exist in the contracts schema. For the POC, **set `enable_graph=False` by default** and remove the graph-related logic. The `config["graph"] = True` line should be conditional.

Specifically, in the `build()` method:
- Change `self.build(force=force, enable_graph=enable_graph, graph_mode=graph_mode)` calls to use `enable_graph=False`
- In the CLI `build` command (see Step 8), set `enable_graph=False` regardless of `--graph-mode`

**Alternatively**, if you want to keep graph as a no-op for future use: keep the parameter but never set `config["graph"] = True` when dealing with contracts, since there are no structural edges.

#### 6f. `_compute_structural_relationships()` (lines 165-202)

**This function is message-specific** (same_conversation, sender_receiver, same_sender, same_company, same_label). It references `open_kfid`, `external_userid`, `servicer_userid`, `company`, `label` — fields that don't exist in contracts.

**Action:** You can either:
- **Delete it entirely** (simplest — contracts don't have conversation structure)
- **Leave it but don't call it** (safe — dead code that won't break anything)
- **Replace with contract-specific relationships** (e.g., same_department, same_risk_level) — optional, future enhancement

**Recommendation:** Delete it. It's cleaner. The graph expansion and relationships are chat-specific concepts.

#### 6g. `Searcher.search()` — Add `filters` parameter (lines 815-931)

The original `search()` accepts `label_filter`, `date_from`, `date_to` — these are message-specific. You need to add a **generic `filters` dict** parameter that supports:
- `risk_level`: str or list[str] (e.g., "high" or ["high", "medium"])
- `amount_min`: float (e.g., 5000000)
- `amount_max`: float
- `audit_status`: str or list[str]
- `legal_approval`: bool
- `overruled`: bool
- `department`: str
- `sign_date_from` / `sign_date_to`: str (ISO date)
- `tags`: str or list[str]

**Modify the `_filter` inner function** (lines 853-862) to use these filters:

```python
def search(
    self,
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    expand: bool = True,
    graph_expand: int = 0,
    label_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    use_rerank: bool = True,
    filters: Optional[Dict[str, Any]] = None,  # NEW PARAMETER
) -> List[Dict]:
    ...
    filters = filters or {}

    def _filter(item: Dict) -> bool:
        meta = item.get("metadata", {})

        # Legacy message filters (keep for backward compat)
        if label_filter and meta.get("label") != label_filter:
            return False
        send_time = meta.get("send_time", "")
        if date_from and send_time and str(send_time) < date_from:
            return False
        if date_to and send_time and str(send_time) > date_to:
            return False

        # NEW: Contract metadata filters
        # risk_level filter
        if "risk_level" in filters:
            allowed = filters["risk_level"]
            if isinstance(allowed, str):
                if meta.get("risk_level") != allowed:
                    return False
            elif isinstance(allowed, list):
                if meta.get("risk_level") not in allowed:
                    return False

        # amount_min filter
        if "amount_min" in filters:
            amount = meta.get("amount")
            if amount is None:
                return False
            try:
                if float(amount) < float(filters["amount_min"]):
                    return False
            except (ValueError, TypeError):
                return False

        # amount_max filter
        if "amount_max" in filters:
            amount = meta.get("amount")
            if amount is None:
                return False
            try:
                if float(amount) > float(filters["amount_max"]):
                    return False
            except (ValueError, TypeError):
                return False

        # audit_status filter
        if "audit_status" in filters:
            allowed = filters["audit_status"]
            if isinstance(allowed, str):
                if meta.get("audit_status") != allowed:
                    return False
            elif isinstance(allowed, list):
                if meta.get("audit_status") not in allowed:
                    return False

        # legal_approval filter (bool)
        if "legal_approval" in filters:
            if meta.get("legal_approval") != filters["legal_approval"]:
                return False

        # overruled filter (bool)
        if "overruled" in filters:
            if meta.get("overruled") != filters["overruled"]:
                return False

        # department filter
        if "department" in filters:
            allowed = filters["department"]
            if isinstance(allowed, str):
                if meta.get("department") != allowed:
                    return False
            elif isinstance(allowed, list):
                if meta.get("department") not in allowed:
                    return False

        # sign_date range
        sign_date = meta.get("sign_date", "")
        if "sign_date_from" in filters:
            if sign_date and str(sign_date) < str(filters["sign_date_from"]):
                return False
        if "sign_date_to" in filters:
            if sign_date and str(sign_date) > str(filters["sign_date_to"]):
                return False

        # tags filter (contract must have ALL specified tags, or ANY)
        if "tags" in filters:
            meta_tags = meta.get("tags", [])
            if isinstance(meta_tags, str):
                try:
                    meta_tags = json.loads(meta_tags)
                except (json.JSONDecodeError, TypeError):
                    meta_tags = []
            if not isinstance(meta_tags, list):
                meta_tags = [str(meta_tags)]
            required_tags = filters["tags"]
            if isinstance(required_tags, str):
                required_tags = [required_tags]
            # Require ALL tags to match
            if not all(tag in meta_tags for tag in required_tags):
                return False

        return True
```

**Important:** The `_filter` function is called in two places within `search()` — make sure to pass `filters` to both call sites. Also update `_graph_expand()` to accept and check `filters` (or simply skip graph expansion since we're disabling graphs for contracts).

#### 6h. `_format_results()` (lines 1063-1092)

**Change the column headers and metadata fields** displayed in CLI results. The original shows: `#`, `ID`, `Score`, `From`, `Label`, `Content`, `Info`. Change to contract-relevant columns:

```python
rows.append([
    i,
    r["id"][:25],
    f"{r.get('score', 0):.4f}",
    str(meta.get("title", "") or f"Contract #{meta.get('contract_id', '-')}")[:12],
    str(meta.get("risk_level", "-")),
    text_preview,
    f"¥{meta.get('amount')}" if meta.get("amount") else "-",
])
```

And update headers:
```python
headers=["#", "ID", "Score", "Title", "Risk", "Content", "Amount"],
```

#### 6i. `SYNTHETIC_TEST_QUERIES` and `TEST_QUERIES` (lines 1030-1101)

**Replace entirely** with contract-related test queries:

```python
SYNTHETIC_TEST_QUERIES = [
    {"query": "租资合同租金超過 500 萬", "expected_keys": ["risk_level"], "description": "high-risk rental contract query"},
    {"query": "法务未审批的合同", "expected_keys": ["legal_approval"], "description": "unapproved contract query"},
    {"query": "采购协议设备采购", "expected_keys": ["department"], "description": "procurement contract query"},
    {"query": "overruled 合同", "expected_keys": ["overruled"], "description": "overruled contract query"},
]

TEST_QUERIES = [
    {"query": "租赁合同", "expected_ids": [], "description": "rental contract"},
    {"query": "采购协议", "expected_ids": [], "description": "procurement agreement"},
    {"query": "软件授权", "expected_ids": [], "description": "software license"},
    {"query": "服务外包", "expected_ids": [], "description": "outsourcing service"},
]
```

#### 6j. CLI `search` command (lines 1135-1190)

**Add new CLI options** for contract filters. The original has `--label`, `--date-from`, `--date-to`. Add:
- `--risk-level` (multiple choice: high/medium/low/all)
- `--amount-min` (float)
- `--amount-max` (float)
- `--audit-status` (string)
- `--legal-approval` / `--no-legal-approval` (flag)
- `--overruled` / `--no-overruled` (flag)
- `--department` (string)

**Remove:** The `--agentic` and `--api-base`/`--api-key`/`--model` options can stay (they're for query expansion which still makes sense for contract search). But Onyx Chat references must be removed.

#### 6e. CLI `build` command (lines 1118-1132)

Add `--graph-mode off` default (or make it a no-op):
```python
@cli.command("build")
@click.option("--force", is_flag=True)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
@click.option("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int)
def build_cmd(force, index_path, chunk_size):
    try:
        # Contracts: no graph (no conversation structure)
        builder = IndexBuilder(index_path, chunk_size=chunk_size)
        embeddings = builder.build(force=force, enable_graph=False)
        click.echo(f"✅ 索引就绪 — {embeddings.count()} 个块 | 图: ❌")
    except Exception as e:
        logger.exception("构建失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
```

Remove `--graph-mode` option entirely (or keep it but ignore it).

---

### Step 7: Create `apps/oa/app.py` (Streamlit UI — REWRITE)

**This is a full rewrite.** You cannot simply adapt the corpchat-rag app.py — it's 568 lines of WeChat message analytics. The new app.py should be a focused contract risk screening dashboard.

#### 7a. Remove ALL Onyx Chat references
- Delete the "Onyx Chat" page entirely
- Delete the iframe embed: `st.components.v1.iframe(iframe_url, height=600, scrolling=True)`
- Delete `OLLAMA_URL` import from `core.config`
- Delete the `generate_answer_litellm()` function (or keep it but don't wire it to a UI page)
- Delete the "deprecated" caption about Onyx Chat
- Ensure NO reference to "onyx", "Onyx", "Onyx Chat" remains anywhere in the file

#### 7b. New page structure

```python
# Pages: "Contract Search", "Contract Browser", "Dashboard"
page = st.radio("Navigate", ["Contract Search", "Contract Browser", "Dashboard"], ...)
```

**Page 1 — Contract Search:**
- Search box (text input)
- Filter sidebar panel with:
  - Risk Level dropdown: All / High / Medium / Low (multi-select)
  - Amount range: min/max number inputs
  - Audit Status: multi-select dropdown (Pending / Audited / Approved / Rejected / All)
  - Legal Approval toggle (switch)
  - Overruled toggle (switch)
  - Department: multi-select dropdown or text input
  - Tags: text input (comma-separated)
  - Search mode: keyword / semantic / hybrid (radio)
  - Top-k slider (1-50)
  - Reranker toggle
- Results displayed as a **table** (pandas DataFrame) with columns: ID, Title, Risk Level, Amount, Audit Status, Legal Approval, Sign Date, Score
- Click a row to expand and see contract content in a modal/expander

**Page 2 — Contract Browser:**
- Table of all contracts with same filtering UI
- Pagination or infinite scroll
- Export to CSV button

**Page 3 — Dashboard:**
- Metrics: Total Contracts, High Risk Count, Pending Audit Count, Overruled Count
- Bar chart: Contracts by Risk Level
- Bar chart: Contracts by Department
- Pie chart: Contracts by Audit Status

#### 7c. Import changes

Change all imports from `apps.corpchat.search` → `apps.oa.search`:
```python
from apps.oa.search import (
    load_index,
    Searcher,
    DEFAULT_INDEX_PATH,
    QueryExpander,
    Reranker,
    AgenticDecider,
)
```

Change `from apps.corpchat.agent import (` → **DELETE** this import entirely (on't need the agentic layer — no Onyx Chat). If you want to keep a simple greeting/intent gate, build it inline.

Change `from core.db import get_db_connection` → keep (it's now MySQL-based)
Change `from core.config import OLLAMA_URL, RAG_MODEL` → delete `OLLAMA_URL` (Onyx Chat removed), keep only what you need

#### 7d. Filter panel design

Use `st.sidebar` (or a top-of-page expander) to house filters. Collect filter values into a `filters` dict and pass to `searcher.search(query, ..., filters=filters)`.

```python
def _build_filters():
    """Build a filters dict from Streamlit sidebar widgets."""
    filters = {}

    risk_levels = st.sidebar.multiselect(
        "Risk Level", ["high", "medium", "low"], default=[]
    )
    if risk_levels:
        filters["risk_level"] = risk_levels

    amount_min = st.sidebar.number_input("Min Amount", min_value=0, value=None, placeholder="No minimum")
    amount_max = st.sidebar.number_input("Max Amount", min_value=0, value=None, placeholder="No maximum")
    if amount_min is not None and amount_min > 0:
        filters["amount_min"] = float(amount_min)
    if amount_max is not None and amount_max > 0:
        filters["amount_max"] = float(amount_max)

    audit_statuses = st.sidebar.multiselect(
        "Audit Status", ["pending", "audited", "approved", "rejected"], default=[]
    )
    if audit_statuses:
        filters["audit_status"] = audit_statuses

    if st.sidebar.checkbox("Only Legal Approved", value=False):
        filters["legal_approval"] = True

    if st.sidebar.checkbox("Only Overruled", value=False):
        filters["overruled"] = True

    departments = st.sidebar.text_input("Department (comma-separated)", placeholder="e.g. Legal, Procurement")
    if departments:
        dept_list = [d.strip() for d in departments.split(",") if d.strip()]
        filters["department"] = dept_list if len(dept_list) > 1 else dept_list[0]

    return filters
```

**Verify against your real schema:** The risk_level values, audit_status values, and department names must match what's in your MySQL `contracts` table. Run `SELECT DISTINCT risk_level FROM contracts;` and `SELECT DISTINCT audit_status FROM contracts;` to get the actual values.

#### 7e. Results table

```python
def _render_results_table(results, raw_hits):
    """Render search results as a table with expandable contract content."""
    if not raw_hits:
        st.info("No results found.")
        return

    # Build DataFrame for display
    rows = []
    for doc in raw_hits:
        meta = doc.get("metadata", {})
        rows.append({
            "ID": meta.get("contract_id", doc.get("id", "")),
            "Title": meta.get("title", "")[:80],
            "Risk": meta.get("risk_level", "-"),
            "Amount": f"{meta.get('amount', 0):,.2f}" if meta.get("amount") else "-",
            "Audit Status": meta.get("audit_status", "-"),
            "Legal Approval": "✅" if meta.get("legal_approval") else "❌",
            "Overruled": "⚠️" if meta.get("overruled") else "–",
            "Sign Date": meta.get("sign_date", "-"),
            "Score": f"{doc.get('score', 0):.4f}",
        })
    df = pd.DataFrame(rows)

    # Display table — allow selection for detail view
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Expandable detail view
    for doc in raw_hits:
        meta = doc.get("metadata", {})
        with st.expander(f"📄 {meta.get('title', 'Contract')} — Risk: {meta.get('risk_level', 'N/A')}"):
            st.markdown(f"**Contract ID:** {meta.get('contract_id', 'N/A')}")
            st.markdown(f"**Amount:** {meta.get('amount', 'N/A')}")
            st.markdown(f"**Department:** {meta.get('department', 'N/A')}")
            st.markdown(f"**Audit Status:** {meta.get('audit_status', 'N/A')}")
            st.markdown(f"**Legal Approval:** {'Yes' if meta.get('legal_approval') else 'No'}")
            st.markdown(f"**Overruled:** {'Yes' if meta.get('overruled') else 'No'}")
            st.markdown("---")
            st.markdown(doc.get("text", "")[:3000] + "...")
```

---

### Step 8: Create `apps/oa/build_index.py`

**Base file:** `/corpchat-rag/apps/corpchat/build_index.py`

**Changes required:**
- Line 5-12: Update docstring to mention MySQL contracts, not PostgreSQL messages
- Line 16: Change "psycopg2" → "pymysql"
- Line 33: Change `from core.config import DB_CONFIG` (same import, just the config now has MySQL)
- Line 73-80: The `_build_via_search_module()` function imports `search.py` — this is fine, just update the path from `apps/corpchat/` → `apps/oa/`. But since you're creating `apps/oa/search.py` as a standalone file (not importing from corpchat-rag), update the import path.
- Lines 89-158: `_build_legacy()` — this is deprecated/legacy and references `contacts` and `messages` tables. **Delete this function** — it's not relevant to contracts.
- Line 213: Change `"corpchat_index"` → `"oa_index"` or use the same path as `DEFAULT_INDEX_PATH`
- The CLI in `search.py` (Step 6) is the primary build entry point. This `build_index.py` is a secondary wrapper — keep it for backward compat but simplify.

---
---

### Step 11: Create `PROJECT_OVERVIEW.md`

Write a comprehensive project overview document at `/Users/ivanlee/Desktop/oa-rag/PROJECT_OVERVIEW.md` with:
- Project goal and background
- Data model (MySQL table structure — use your real schema)
- Core features list
- Tech stack
- How to build the index
- How to run search (CLI + Web UI)
- Current status (POC) and next steps

---

### Step 12: Create `.gitignore`

**Base file:** `/corpchat-rag/.gitignore`

**Changes required:**
- Change `apps/corpchat/search_index/` → `apps/oa/search_index/`
- Change `models/` stays the same
- Everything else stays

---

## 4. Key Design Decisions to Make (Answer These)

Before implementing, verify these against your real MySQL table:

| Question | Your Action |
|----------|-------------|
| What are the exact column names and types in `contracts`? | `DESCRIBE contracts;` |
| What values does `risk_level` contain? | `SELECT DISTINCT risk_level FROM contracts;` |
| What values does `audit_status` contain? | `SELECT DISTINCT audit_status FROM contracts;` |
| Is `legal_approval` BOOLEAN/TINYINT/ENUM? | `DESCRIBE contracts;` — check type |
| Is `overruled` BOOLEAN/TINYINT/ENUM? | `DESCRIBE contracts;` — check type |
| Is `tags` JSON/LONGTEXT/VARCHAR? | `DESCRIBE contracts;` — check type |
| Is `amount` DECIMAL/INT/VARCHAR? | `DESCRIBE contracts;` — check type |
| Is `content` TEXT/LONGTEXT? | `DESCRIBE contracts;` — check type |
| How many contracts are in the table? | `SELECT COUNT(*) FROM contracts;` |
| Is the content in Chinese, English, or bilingual? | `SELECT content FROM contracts LIMIT 1;` |

---

## 5. Files NOT to Copy / Not to Modify

- `apps/corpchat/agent.py` — **DO NOT COPY**. This is the Onyx Chat agentic layer. Onyx Chat is being completely removed.
- `apps/corpchat/ingest.py` — **DO NOT COPY**. This uploads data to Onyx's external API. Not relevant.
- `apps/corpchat/visualize_graph.py` — **DO NOT COPY** (if it exists). Graph visualization is for conversation graphs.
- `apps/corpchat/run_agent.py` — **DO NOT COPY**. Onyx Chat agent runner.
- `apps/corpchat/test_search.py` — **DO NOT COPY** or **adapt** if needed.
- `apps/corpchat/gen_fake_data.py` — **DO NOT COPY**. This was an early/legacy data generator. Use `gen_fake_msg.py` as the template instead for `gen_fake_contract.py`.
- `tests/` directory — **OPTIONAL**. Only copy if you want to adapt the test framework. Tests in corpchat-rag test the `apps.corpchat.app` module and `apps.corpchat.search` — they'd need significant rewriting for oa-rag.

---

## 6. Testing Checklist

After implementation, verify:

1. **Config loads:** `python -c "from core.config import DB_CONFIG; print(DB_CONFIG)"` → shows MySQL config
2. **DB connection works:** `python -c "import pymysql; conn=pymysql.connect(**DB_CONFIG); print('OK'); conn.close()"`
3. **Contracts fetch works:** `python -c "from apps.oa.search import IndexBuilder; b=IndexBuilder(); print(len(b._fetch_contracts()), 'contracts')"` → should return >0
4. **Index builds:** `cd /oa-rag && python apps/oa/search.py build --force` → should create index with contract chunks
5. **Basic search works:** `python apps/oa/search.py search "租赁" --mode hybrid` → returns results
6. **Filter search works:** `python apps/oa/search.py search "采购" --risk-level high --amount-min 1000000` → filtered results
7. **Streamlit UI loads:** `streamlit run apps/oa/app.py` → no errors, no Onyx Chat page
8. **Onyx Chat is gone:** Search for "onyx" or "Onyx" in all `.py` files in `oa-rag/` → zero results
9. **All pages render:** Contract Search, Contract Browser, Dashboard — all show data correctly

---

## 7. Important Notes

- **Encoding:** The corpchat-rag codebase has extensive Chinese comments and docstrings. Keep the same Chinese+English bilingual comment style for consistency. The search framework was designed for bilingual (Chinese+English) WeCom messages — it should work fine for bilingual contract content too.
- **jieba segmentation:** The `_segment()` function uses jieba for Chinese tokenization. Keep it — jieba passes through English text unchanged, so it won't harm English contracts.
- **txai version:** The project uses `txtai==9.12.0`. The `filters` dict approach I described uses **post-filtering** (filter after search returns results) rather than txtai's native `where` clause. This is simpler and more flexible. If you want to use txtai's native filtering instead, you'd use the `where` parameter in `embeddings.search()`, but the post-filter approach is recommended for POC.
- **Graph features:** Disable entirely for contracts. The structural relationship computation is message-specific. You can re-enable contract-specific graph features (e.g., same_department, same_risk_level) in a future phase.
- **The original search.py has 1292 lines** — most of the file (QueryExpander, Reranker, RRF fusion, CLI, synthetic tests) can be **kept as-is**. Only `IndexBuilder` (the `_fetch_messages` and `_chunk_message` methods) and `Searcher.search` (adding `filters` param) need significant changes.

---

## 8. Quick Reference: CorpChat Code Map

```
search.py structure (1292 lines):
├── Constants & config (lines 33-125)      → Change psycopg2→pymysql, DB_CONFIG defaults
├── Helper functions (lines 134-202)        → _clean_text_from_enriched (keep), _segment (keep), 
├──                                             _compute_structural_relationships (DELETE or stub)
├── IndexBuilder (lines 210-415)            → _fetch_messages→_fetch_contracts, _chunk_message→_chunk_contract
├── QueryExpander (lines 423-520)           → KEEP AS-IS (works for any domain)
├── Reranker (lines 528-576)                → KEEP AS-IS
├── Searcher (lines 585-947)                → Add filters param to search(), update _filter()
├── load_index (lines 954-960)              → KEEP AS-IS
├── AgenticDecider (lines 967-1023)          → KEEP AS-IS (for CLI --agentic mode)
├── Synthetic test data (lines 1030-1055)   → Replace with contract queries
├── _format_results (lines 1063-1092)       → Change column headers/fields for contracts
├── TEST_QUERIES (lines 1095-1101)          → Replace with contract queries
├── CLI commands (lines 1111-1292)          → Add --risk-level, --amount-min, etc.; remove --graph-mode
└── __main__ (line 1291)                    → KEEP
```

Good luck! The core search infrastructure is solid — most of the work is swapping the data source and adapting the UI. **Read every file before modifying it.**