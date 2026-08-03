# 01 — Chinese-capable hybrid base

**What to build:** The retrieval base works natively on Chinese. A user searching `物流報價 方案` gets the message actually about logistics quotation, not merely any message containing 方案. A user searching `投資美國債券跟藍籌股` gets the message containing those keywords, not keywordless noise. A bare label search (e.g. `product_inquiry`) no longer ranks all documents of that label. Label filtering still scopes results. The search UI shows clean message content. A fresh clone installs and runs the search correctly with no committed secrets and no stale machine-specific index.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Chinese word segmentation (jieba `cut_for_search`) applied to BM25 at both index and query time
- [ ] Embedding model swapped to `BAAI/bge-m3` (production default, local-model-path support)
- [ ] Match surface is content + `customer_name (company)` title only; label/time/sender/receiver/msgid are structured metadata (filter/display/LLM-context only); the `\n---\nMetadata:` glue and reverse-parsing removed
- [ ] Hybrid fusion fetches each document by its actual document id, not by re-searching `id:xxx` as text
- [ ] Reranker disabled by default (English-only model must not demote Chinese results)
- [ ] Stale machine-specific `search_index/` removed from repo and git-ignored
- [ ] Hardcoded API keys and `key.txt` removed; keys required from environment; `.env`/`models/`/`search_index/` git-ignored
- [ ] `requirements.txt` corrected with real search-core deps (`txtai`, `chonkie`, `tabulate`, `sentence-transformers`, `jieba`, `pytest`)
- [ ] pytest regression suite (deterministic in-memory index from conversation templates, bge-m3) green, asserting: 物流報價 方案 → logistics-quotation message; 投資美國債券跟藍籌股 → keyword-containing message; bare label search does not rank all label docs; label filter scopes correctly