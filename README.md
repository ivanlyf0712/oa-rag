# OA RAG

**繁體中文**：OA RAG 是一個用於企業合約風險篩查與語義檢索的系統，支援混合式關鍵字/向量搜尋、查詢擴展、RRF 融合與 reranking，並以合約資料為核心。

**English**: OA RAG is a contract risk screening and semantic retrieval system for enterprise use. It supports hybrid keyword/vector search, query expansion, RRF fusion, and reranking, with contract records as the primary data source.

---

## 功能特色 / Features

- **合約風險篩查**：針對合約欄位與內容進行風險提示與檢索。
- **Hybrid Search**：結合關鍵字搜尋與語意向量搜尋。
- **Query Expansion**：支援查詢擴展與多路召回。
- **RRF Fusion**：透過 Reciprocal Rank Fusion 彙整多個查詢結果。
- **Reranking**：可選擇使用 reranker 提升結果品質。
- **資料可追溯**：保留原始欄位，並加入解碼後的標籤與上下文資訊。
- **統一 Agentic UI**：以自然語言提問，LangChain 代理自動選擇合約搜尋或風險搜尋工具，
  並顯示意圖、工具與路由過程。
- **Streamlit UI**：提供合約搜尋、瀏覽與風險摘要介面。

- **Contract risk screening**: identify and search for risk signals in contract fields and text.
- **Hybrid search**: combine keyword search with semantic vector search.
- **Query expansion**: support multiple query variants for better recall.
- **RRF fusion**: merge results from multiple retrieval paths using Reciprocal Rank Fusion.
- **Reranking**: optional reranker support for improved ranking quality.
- **Traceable data**: preserve raw fields while adding decoded labels and contextual metadata.
- **Unified agentic UI**: ask in natural language; a LangChain agent automatically
  chooses the contract-search or risk-search tool and shows intent, tool, and routing.
- **Streamlit UI**: provides contract search, browsing, and risk summary views.

---

## 專案結構 / Project Structure

- `apps/`：OA 主應用與介面程式碼 / OA app and UI code（`apps/search/` 檢索與代理引擎）
- `core/`：資料庫連線與設定 / DB connection and runtime config
- `data/`：資料說明、欄位對照與匯出文件 / data notes, field mappings, and exports
- `docs/`：設計文件、索引說明與專案背景 / design docs, index notes, and project background
- `lib/`：前端共用元件庫（歷史遺留）/ shared front-end libraries (legacy)
- `scripts/`：檢查與維運腳本 / validation and ops scripts（索引重建 build_index.py、附件同步 sync_attachments.py、本地啟動 run_streamlit.sh）
- `tests/`：自動化測試 / automated tests（`tests/legacy_corpchat/` 為 CorpChat 遺留測試，預設排除）
- `uploads/`：已上載合約附件 / uploaded contract attachments
- 部署 / deployment：`Dockerfile`、`docker-compose.yml`、`Makefile`、`.env.example`

---

## 快速開始 / Quick Start

### 方式 A：Docker 一鍵啟動（推薦）/ Docker one-command stack (recommended)

參考 corpchat-rag 的做法，整個堆疊（MySQL 8 + Streamlit app）已容器化：
Mirroring corpchat-rag, the whole stack (MySQL 8 + Streamlit app) is containerized:

```bash
cp .env.example .env      # 填入 DB_* / LITELLM_API_KEY 等 / fill in secrets
make up                   # 校驗 .env 並 docker compose up -d --build
make db-import            # 首次：把本機 MySQL 的資料匯入容器 / one-time: import host MySQL data
docker compose ps         # 查看狀態；UI 在 http://localhost:8501
```

其他常用 target / other targets：`make logs`、`make down`（保留資料卷）、`make index`（重建索引）、`make test`。

資料持久化 / persistence：四個 named volumes —— `oa-mysql-data`（資料庫）、`oa-uploads`（附件）、
`oa-index`（搜尋索引）、`hf-cache`（嵌入模型快取，首次約 3.5G 下載）。
注意：容器內 MySQL 對外埠號是 **3307**（本機 3306 已被佔用）/ the dockerized MySQL is published on host port 3307.

### 方式 B：本機開發 / Local development

```bash
# 1) 安裝相依套件 / Install dependencies
python3 -m venv venv && venv/bin/pip install -r requirements.txt

# 2) 設定環境變數 / Configure environment variables
cp .env.example .env      # 填入資料庫、模型與檢索服務設定 / fill in DB, model, LLM settings

# 3) 啟動應用 / Run the app
scripts/run_streamlit.sh  # 或 / or: venv/bin/python -m streamlit run apps/app.py

# 4) 執行測試 / Run tests
venv/bin/python -m pytest tests/ --ignore=tests/legacy_corpchat -q   # 或 / or: make test

# 5) 重建索引 / Rebuild the search index
venv/bin/python scripts/build_index.py --force
```


---

## 搜尋與索引 / Search and Indexing

此專案使用 MySQL 合約資料作為索引來源，並搭配語意嵌入與查詢擴展來提升召回率。索引與搜尋的細節可以參考：

- `apps/search/_core.py`：txtai 混合索引（BM25 + 向量 + RRF）與 `Searcher.build()`
- `scripts/build_index.py`：索引重建入口 / index build entry point
- `docs/`：索引與欄位設計文件 / index and field design docs

This project uses MySQL contract records as the index source and combines semantic embeddings with query expansion to improve recall. See:

- `apps/search/_core.py` — txtai hybrid index (BM25 + vector + RRF) and `Searcher.build()`
- `scripts/build_index.py` — index build entry point
- `docs/` — index and field design docs

---

## 統一 Agentic UI / Unified Agentic UI

自從 LangChain agentic 改版後，應用只有一個搜尋入口：左側邊欄的 **「Ask (Agentic)」** 檢視。
你以自然語言提問，代理會自動決定要用哪一種搜尋工具：

- **合約搜尋 (contract_search)**：一般合約查詢（預設），回傳答案與佐證合約。
- **風險搜尋 (risk_search)**：風險意圖查詢（例如「show contracts where risk was not accepted」），
  回傳風險評分表與逐約摘要。
- **澄清 (clarify)**：當問題模糊時，代理會先反問澄清而不呼叫任何工具。

畫面上方會顯示三項路由中繼資料（偵測到的意圖、使用的工具、路由方式），並可在
「Agent steps」展開檢視完整的決策過程。若 LangChain 模型提供者（Ollama）無法連線，
系統會自動退回確定性的內建路由（fallback），不會導致搜尋失敗。

Since the LangChain agentic rework, the app has a single search entry point: the
**“Ask (Agentic)”** view in the sidebar. You ask in natural language and the agent decides
which search tool to use:

- **contract_search**: general contract questions (the default), returning an answer plus
  supporting contract evidence.
- **risk_search**: risk-intent questions (e.g. “show contracts where risk was not accepted”),
  returning a risk-scored table with per-contract summaries.
- **clarify**: when a question is ambiguous, the agent asks a clarifying question instead of
  calling any tool.

The top of the view surfaces three routing metadata fields (detected intent, tool used, and
routing mode), and the “Agent steps” expander shows the full decision trace. If the LangChain
model provider (Ollama) is unreachable, the app automatically falls back to a deterministic
built-in router so search never fails.

### 組態 / Configuration

代理透過既有的 LiteLLM 相容環境變數連接本地 Ollama 模型：

The agent connects to a local Ollama model via the existing LiteLLM-compatible env vars:

| 變數 / Variable | 預設 / Default | 說明 / Description |
| --- | --- | --- |
| `LITELLM_BASE_URL` | `https://litellm.dchbi.app/` | LiteLLM 相容代理位址 / LiteLLM-compatible proxy URL |
| `LITELLM_MODEL` | `dseek-v4-flash` | 用於工具呼叫的模型 / Model used for tool calling |

### 相依套件 / Dependencies

新版代理層使用最小化、與供應商無關的 LangChain 套件（見 `requirements.txt`）：

The agentic layer uses a minimal, provider-agnostic LangChain set (see `requirements.txt`):

- `langchain-core` — 工具 (`@tool`) 與聊天模型抽象 / tool and chat-model abstractions
- `langchain-ollama` — 本地工具呼叫模型提供者 (`ChatOllama`) / local tool-calling provider

相關程式碼 / Relevant code: `apps/search/langchain_agent.py`（代理 agent）、`apps/search_cli.py`（`build_contract_tool` / `build_risk_tool`）、`apps/app.py`（統一 UI unified UI）。

---

## 說明 / Notes

- 舊的 CorpChat 相關路徑與歷史資料可能仍保留在倉庫中，僅作參考用途。
- OA 合約篩查是目前推薦的主線入口。
- 若你要修改索引、欄位解碼或搜尋流程，建議先閱讀 `apps/search/_core.py` 與 `apps/search/langchain_agent.py`。

- Older CorpChat paths and historical data may still exist in the repository for reference only.
- OA contract screening is the recommended primary entry point.
- If you need to change indexing, field decoding, or search flow, start with `apps/search/_core.py` and `apps/search/langchain_agent.py`.

---

## 重建環境 / Recreate This Environment

> 真實資料（MySQL dump、CSV、uploads、search index、`.env`）**永不提交**。
> Real data (MySQL dumps, CSVs, uploads, the search index, `.env`) is **never
> committed**. Follow these steps to recreate a working copy from git alone.

```bash
# 1. Clone & install / 複製並安裝
git clone <this-repo> && cd oa-rag
python -m venv venv && venv/bin/pip install -r requirements.txt

# 2. Configure secrets / 設定密鑰
cp .env.example .env   # fill DB_*, LITELLM_API_KEY, LITELLM_BASE_URL

# 3. Generate FAKE demo data / 產生假的示範資料
venv/bin/python scripts/gen_fake_contracts.py          # sql + csv + uploads
mysql -u <user> -p <db> < data/fake_seed.sql           # seed MySQL

# 4. Build the search index / 建立搜尋索引
venv/bin/python scripts/build_index.py --force

# 5. Run / 啟動
venv/bin/streamlit run apps/app.py                     # local
# or Docker: make up && make db-import && make index
```

- 測試不需要真實資料 / Tests need no data: `venv/bin/python -m pytest` (197 tests, fakes only).
- 假資料產生器 / Generator options: `--count N --seed S --sql|--csv|--uploads`
  (deterministic; exercises Over5M / risk flags / statuses / attachments).
- 真實資料還原 / To restore REAL data instead: import your own
  `formtable_main_385` dump into MySQL, drop files into `uploads/contracts/`,
  then run step 4.

---

## 授權 / License

請依照專案實際授權條款處理。 / Please follow the repository’s actual license terms.
