# OA-RAG Project Overview

OA-RAG is a contract risk screening and retrieval system built on txtai hybrid search, query expansion, RRF fusion, and reranking. It replaces the prior chat-search domain with contract-centric search and filtering over MySQL contract records.

## Data model
- Source table: 
- Primary key: uid=1000(ivanleeyf) gid=1000(ivanleeyf) groups=1000(ivanleeyf),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),100(users),1001(docker)
- Core searchable fields include contract reference/title, counterparty, product/services, contract amount, contract dates, department, approval flags, risk prompts, and long-form draft/final contract text.

## Core features
- Hybrid keyword + semantic search
- RRF fusion across multiple query variants
- Optional reranking
- Contract metadata filtering
- Streamlit contract search, browser, and dashboard
- No Onyx Chat or embedded chat UI

## Build and run
- Install: 
- Build index: 
- Search CLI: 
- Streamlit: 

## Notes
- Graph features are disabled for contracts.
- MySQL is used via .
