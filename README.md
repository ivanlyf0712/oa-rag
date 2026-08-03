# CorpChat RAG

企业微信对话智能搜索与 RAG 系统。基于 txtai 混合搜索 + LLM 查询扩展 + 交叉编码器重排序的 Onyx 风格搜索框架。

## 功能特性

- 🔍 **Onyx 风格搜索**: 混合搜索 (BM25 + 向量) + LLM 查询扩展 + 加权 RRF 融合
- 🕸️ **图增强搜索**: 基于 txtai 图的一跳邻居扩展
- ⚡ **交叉编码器重排序**: 使用 cross-encoder/ms-marco-MiniLM-L-6-v2
- 💬 **Streamlit 交互界面**: 联系人、消息、聊天记录、语义搜索一体化
- 🤖 **RAG 问答**: 基于 LiteLLM 的自然语言答案生成

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据库

编辑 `core/config.py` 或设置环境变量:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=invoices
export DB_USER=ocr
export DB_PASSWORD=ocrpass
```

### 3. 生成测试数据

```bash
python apps/corpchat/gen_fake_msg.py
```

### 4. 构建搜索索引

```bash
python apps/corpchat/search.py build --force
```

### 5. 启动 Streamlit 应用

```bash
streamlit run apps/corpchat/app.py
```

## 搜索 CLI

```bash
# 构建索引
python apps/corpchat/search.py build --force --graph-mode auto

# 搜索 (全链路模式)
python apps/corpchat/search.py search "诈骗" --mode hybrid --expand --rerank

# 合成测试基准
python apps/corpchat/search.py synthetic-benchmark
```

## 项目结构

```
corpchat-rag/
├── apps/corpchat/
│   ├── app.py                # Streamlit 交互界面
│   ├── search.py             # 搜索核心引擎 (IndexBuilder, Searcher, Reranker, etc.)
│   ├── build_index.py        # 索引构建脚本
│   ├── gen_fake_msg.py       # 测试数据生成
│   ├── ingest.py             # 数据导入
│   └── search_index/         # 预构建的 txtai 索引
├── core/
│   ├── config.py             # 数据库与 API 配置
│   ├── db.py                 # 数据库连接
│   └── embedding.py          # 嵌入工具
├── requirements.txt
└── README.md