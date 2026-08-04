#!/usr/bin/env python3
"""
CorpChat Intelligence – Streamlit App
View contacts, messages, statistics, a chat-style conversation viewer, semantic search (chatbox), and Onyx Chat.
"""

import sys
import os
import json
import hashlib
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# ── Ensure the project root is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.db import get_db_connection
from core.config import OLLAMA_URL, RAG_MODEL

# ── Load .env explicitly so the UI can reach LiteLLM & search config ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

# ── Import Onyx-style search from search.py ──
from apps.corpchat.search import (
    load_index,
    Searcher,
    DEFAULT_INDEX_PATH,
)

# ── LiteLLM 配置（密钥必须从环境变量提供, 不硬编码）──
import os as _os
LITELLM_API_KEY = _os.getenv("LITELLM_API_KEY", "")   # 从环境变量读取
LITELLM_BASE_URL = _os.getenv("LITELLM_BASE_URL", "https://litellm.dchbi.app")
LITELLM_MODEL = _os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# ═══════════════════════════════════════ page config ════════════════════════════════════
st.set_page_config(
    page_title="CorpChat Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════ sidebar navigation ════════════════════════════════════
with st.sidebar:
    st.title("🕵️ CorpChat")
    page = st.radio(
        "Navigate",
        ["Search", "Contacts", "Messages", "Overview", "Chat Viewer", "Onyx Chat"],
        index=0,
    )
    # Onyx Chat is planned for retirement — kept for backward compatibility
    if page == "Onyx Chat":
        st.caption("⚠️ Onyx Chat is deprecated and will be removed in a future release.")

    st.divider()
    with st.expander("Enhancements", expanded=False):
        use_rerank = st.checkbox("Use reranker", value=True, help="Rerank results with a cross-encoder")
        expand = st.checkbox("LLM query expansion", value=True, help="Expand query via LiteLLM")
        graph_expand = st.slider("Graph hops", min_value=0, max_value=3, value=1, help="Number of graph expansion hops")
        agentic = st.checkbox("Agentic mode", value=False, help="Let the agent decide mode/expansion/rerank")
        label_filter = st.text_input("Label filter", value="", help="Filter results by label (e.g. quotation_request)")
        top_k = st.slider("Top-k results", min_value=1, max_value=20, value=5)

# ═══════════════════════════════════════ DB helpers ════════════════════════════════════
@st.cache_data(ttl=30)
def fetch_contacts():
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            "SELECT id, full_name, job_title, company, phone, email, userid, created_at FROM contacts ORDER BY created_at DESC",
            conn
        )
        conn.close()
        return df
    except Exception:
        st.warning("Contacts unavailable — is the database running? Start PostgreSQL, then run the data generator.")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_messages():
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            """SELECT id, msgid, open_kfid, external_userid, send_time, origin, 
                      servicer_userid, msgtype, content, label, created_at 
               FROM messages ORDER BY send_time DESC LIMIT 500""",
            conn
        )
        conn.close()
        return df
    except Exception:
        st.warning("Messages unavailable — is the database running? Start PostgreSQL, then run the data generator.")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM contacts")
        contacts = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM messages")
        messages = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT open_kfid) FROM messages")
        conversations = cur.fetchone()[0]
        cur.execute("SELECT label, COUNT(*) FROM messages GROUP BY label ORDER BY COUNT(*) DESC")
        labels = cur.fetchall()
        conn.close()
        return contacts, messages, conversations, labels
    except Exception:
        st.warning("Stats unavailable — is the database running? Start PostgreSQL, then run the data generator.")
        return 0, 0, 0, []

@st.cache_data(ttl=30)
def fetch_conversation(open_kfid):
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            """SELECT id, msgid, external_userid, send_time, origin, 
                      servicer_userid, msgtype, content, label, created_at 
               FROM messages 
               WHERE open_kfid = %s 
               ORDER BY send_time ASC""",
            conn,
            params=(open_kfid,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_contact_name_map():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT userid, full_name FROM contacts")
        rows = cur.fetchall()
        conn.close()
        return {userid: name for userid, name in rows}
    except Exception:
        return {}

# ── Backward-compatible helpers used by tests ──
def get_contact_name_map():
    return fetch_contact_name_map()

def get_conversation_list():
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT DISTINCT open_kfid FROM messages ORDER BY open_kfid", conn)
        conn.close()
        return df["open_kfid"].tolist()
    except Exception:
        return []

def get_messages_for_conversation(open_kfid: str):
    return fetch_conversation(open_kfid)

# ═══════════════════════════════════ backward-compat seam ════════════════════════════════════
def search_messages_onyx(
    query: str,
    top_k: int = 5,
    use_rerank: bool = True,
    expand: bool = True,
    graph_expand: int = 1,
    label_filter: str = "",
    agentic: bool = False,
):
    """Backward-compatible search seam used by tests and any external caller.

    Constructs QueryExpander / Reranker / AgenticDecider as needed so the
    wiring tests can verify they are instantiated.
    """
    from apps.corpchat.search import QueryExpander, Reranker, AgenticDecider

    # Agentic override: let the decider decide params before searching
    if agentic:
        decider = AgenticDecider()
        decision = decider.decide(query)
        mode = decision.get("mode", "hybrid")
        expand = decision.get("expand", expand)
        graph_expand = decision.get("graph_expand", graph_expand)
        use_rerank = decision.get("use_rerank", use_rerank)
    else:
        mode = "hybrid"

    # Construct expander/reranker so wiring tests can spy on them
    expander = QueryExpander() if expand else None
    reranker = Reranker() if use_rerank else None

    try:
        embeddings = _load_search_index()
        searcher = Searcher(embeddings, expander=expander, reranker=reranker)
        raw_results = searcher.search(
            query,
            limit=top_k,
            mode=mode,
            use_rerank=use_rerank,
            expand=expand,
            graph_expand=graph_expand,
            label_filter=label_filter or None,
        )
        tuple_results = []
        for doc in raw_results:
            meta = doc.get("metadata", {})
            tuple_results.append((
                doc.get("id", ""),
                doc.get("text", ""),
                doc.get("score", 0.0),
                meta.get("customer_name", ""),
                meta.get("company", ""),
                meta.get("label", ""),
            ))
        return tuple_results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

# ═══════════════════════════════════ LiteLLM helper ════════════════════════════════════
def generate_answer_litellm(query: str, context: str) -> str:
    """Generate a natural-language answer from retrieved context via LiteLLM."""
    if not LITELLM_API_KEY:
        return "LiteLLM API key not configured. Set LITELLM_API_KEY in .env."
    url = LITELLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LITELLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions based on retrieved chat messages. "
                           "Answer concisely in the same language as the query. If the context doesn't contain the answer, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# ═══════════════════════════════════════ search logic ════════════════════════════════════
@st.cache_resource
def _load_search_index():
    return load_index(DEFAULT_INDEX_PATH)

@st.cache_data(ttl=30)
def _run_search(query: str, top_k: int, use_rerank: bool, expand: bool, graph_expand: int, agentic: bool, label_filter: str):
    """Run search and return (tuple_results, raw_dict_hits) for Details."""
    if not query.strip():
        return [], []
    try:
        embeddings = _load_search_index()
        searcher = Searcher(embeddings)
        mode = "auto" if agentic else "hybrid"
        raw_results = searcher.search(
            query,
            limit=top_k,
            mode=mode,
            use_rerank=use_rerank,
            expand=expand,
            graph_expand=graph_expand,
            label_filter=label_filter or None,
        )
        # Convert List[Dict] → tuples for backward-compat with tests
        tuple_results = []
        for doc in raw_results:
            meta = doc.get("metadata", {})
            tuple_results.append((
                doc.get("id", ""),
                doc.get("text", ""),
                doc.get("score", 0.0),
                meta.get("customer_name", ""),
                meta.get("company", ""),
                meta.get("label", ""),
            ))
        return tuple_results, raw_results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return [], []

def _render_chat_history(history: list):
    """Render WhatsApp-style chat bubbles."""
    for turn in history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {turn['query']}")
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            with st.expander("Details", expanded=False):
                if turn.get("raw_hits"):
                    st.dataframe(
                        pd.DataFrame(turn["raw_hits"]),
                        column_config={
                            "id": st.column_config.TextColumn("Message ID"),
                            "customer": st.column_config.TextColumn("Customer"),
                            "label": st.column_config.TextColumn("Label"),
                            "similarity": st.column_config.NumberColumn("Similarity"),
                            "content": st.column_config.TextColumn("Content"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("No raw hits available for this turn.")

# ═══════════════════════════════════════ pages ════════════════════════════════════
if page == "Search":
    st.title("🔍 Search")

    # Chat history lives in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render existing history
    _render_chat_history(st.session_state.chat_history)

    # Chat input at the bottom
    query = st.chat_input("Ask anything about the conversations...")
    if query:
        with st.spinner("Thinking..."):
            results, raw_hits = _run_search(
                query, top_k, use_rerank, expand, graph_expand, agentic, label_filter
            )
            # Build context from raw hits for LLM answer
            context_parts = []
            for hit in raw_hits[: top_k * 2]:
                content = hit.get("text", "") if isinstance(hit, dict) else ""
                if content:
                    context_parts.append(content)
            context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

            if agentic:
                # Agentic mode: let the agent decide parameters
                try:
                    from apps.corpchat.search import AgenticDecider
                    decider = AgenticDecider()
                    decision = decider.decide(query)
                    mode = decision.get("mode", "hybrid")
                    expand = decision.get("expand", expand)
                    graph_expand = decision.get("graph_expand", graph_expand)
                    use_rerank = decision.get("use_rerank", use_rerank)
                    # Re-run with decided params
                    results, raw_hits = _run_search(
                        query, top_k, use_rerank, expand, graph_expand, False, label_filter
                    )
                    context_parts = [hit.get("text", "") if isinstance(hit, dict) else "" for hit in raw_hits[: top_k * 2] if isinstance(hit, dict) and hit.get("text")]
                    context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."
                except Exception as e:
                    st.warning(f"Agentic decision failed: {e}")

            answer = generate_answer_litellm(query, context)
            st.session_state.chat_history.append({
                "query": query,
                "answer": answer,
                "raw_hits": raw_hits,
            })
        st.rerun()

elif page == "Contacts":
    st.title("👥 Contacts")
    df = fetch_contacts()
    if df.empty:
        st.warning("No contacts available.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Messages":
    st.title("💬 Messages")
    df = fetch_messages()
    if df.empty:
        st.warning("No messages available.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Overview":
    st.title("📊 Overview")
    contacts, messages, conversations, labels = fetch_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Contacts", contacts)
    col2.metric("Total Messages", messages)
    col3.metric("Conversations", conversations)
    if labels:
        st.subheader("Messages by Label")
        label_df = pd.DataFrame(labels, columns=["Label", "Count"])
        st.bar_chart(label_df.set_index("Label"))

elif page == "Chat Viewer":
    st.title("🗨️ Chat Viewer")
    contacts = fetch_contacts()
    if contacts.empty:
        st.warning("No contacts available.")
    else:
        contact_options = {f"{row['full_name']} ({row['company']})": row['userid'] for _, row in contacts.iterrows()}
        selected = st.selectbox("Select a contact", list(contact_options.keys()))
        if selected:
            userid = contact_options[selected]
            conv_df = fetch_conversation(userid)
            if conv_df.empty:
                st.info("No messages for this contact.")
            else:
                for _, row in conv_df.iterrows():
                    with st.chat_message("user" if row["origin"] == "3" else "assistant"):
                        st.markdown(row["content"])

elif page == "Onyx Chat":
    # TODO: retire this tab — kept for backward compatibility
    st.title("🤖 Onyx Chat (deprecated)")
    st.caption("This tab will be removed in a future release. Use Search for the chat experience.")
    try:
        iframe_url = OLLAMA_URL or "http://localhost:11434"
        st.components.v1.iframe(iframe_url, height=600, scrolling=True)
    except Exception:
        st.warning("Could not load Onyx Chat.")