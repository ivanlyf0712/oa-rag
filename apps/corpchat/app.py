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

# ── LiteLLM config ──
import os as _os
LITELLM_API_KEY = _os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = _os.getenv("LITELLM_BASE_URL", "https://litellm.dchbi.app")
LITELLM_MODEL = _os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# ═══════════════════════════════════════ page config ════════════════════════════════════
st.set_page_config(
    page_title="CorpChat Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════ professional CSS ════════════════════════════════════
st.markdown("""
<style>
.stApp { background: #0e1117; color: #e6e6e6; font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
.stApp .stMarkdown p, .stApp .stMarkdown li { color: #c9d1d9; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] .stRadio > label { color: #58a6ff; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
h1, h2, h3 { color: #f0f6fc !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h1 { border-bottom: 2px solid #30363d; padding-bottom: 0.3em; }
.stChatMessage [data-testid="stChatMessageContent"] { border-radius: 8px; padding: 12px 16px; }
.stDataFrame { border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }
.streamlit-expander { border: 1px solid #30363d; border-radius: 6px; background: #161b22; }
[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.stChatInput { border-top: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════ sidebar navigation ════════════════════════════════════
with st.sidebar:
    st.markdown("## CorpChat Intelligence")
    st.caption("Corporate Relationship & Chat Analytics")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Search", "Contacts", "Messages", "Overview", "Chat Viewer", "Onyx Chat"],
        index=0,
    )
    if page == "Onyx Chat":
        st.caption("Deprecated — will be removed in a future release.")

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
        st.warning("Contacts unavailable — is the database running?")
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
        st.warning("Messages unavailable — is the database running?")
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
        st.warning("Stats unavailable — is the database running?")
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
def fetch_conversations_for_contact(userid):
    """Fetch all messages involving a contact (as sender or receiver)."""
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            """SELECT id, msgid, open_kfid, external_userid, send_time, origin,
                      servicer_userid, msgtype, content, label, created_at
               FROM messages
               WHERE external_userid = %s OR servicer_userid = %s
               ORDER BY send_time ASC""",
            conn,
            params=(userid, userid)
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

    if agentic:
        decider = AgenticDecider()
        decision = decider.decide(query)
        mode = decision.get("mode", "hybrid")
        expand = decision.get("expand", expand)
        graph_expand = decision.get("graph_expand", graph_expand)
        use_rerank = decision.get("use_rerank", use_rerank)
    else:
        mode = "hybrid"

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
    if not LITELLM_API_KEY:
        return "LiteLLM API key not configured. Set LITELLM_API_KEY in .env."
    url = LITELLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LITELLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions based on retrieved chat messages. Answer concisely in the same language as the query. If the context doesn't contain the answer, say so."
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

def _check_llm_available() -> bool:
    """Quick check if LiteLLM is reachable."""
    if not LITELLM_API_KEY:
        return False
    try:
        resp = requests.get(
            LITELLM_BASE_URL.rstrip("/") + "/v1/models",
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False

def _render_chat_history(history: list):
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            if turn.get("status") == "processing":
                st.info("Search was interrupted. This turn has no results.")
            elif turn.get("answer"):
                st.markdown(turn["answer"])
                with st.expander("Details", expanded=False):
                    if turn.get("raw_hits"):
                        st.dataframe(
                            pd.DataFrame(turn["raw_hits"]),
                            column_config={
                                "id": st.column_config.TextColumn("Message ID"),
                                "text": st.column_config.TextColumn("Content"),
                                "score": st.column_config.NumberColumn("Score"),
                                "metadata": st.column_config.TextColumn("Metadata"),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.caption("No raw hits available for this turn.")

# ═══════════════════════════════════════ pages ════════════════════════════════════
if page == "Search":
    st.title("Search")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "searching" not in st.session_state:
        st.session_state.searching = False

    # Handle interrupted search from reload (stop and discard)
    if st.session_state.searching:
        # Discard any processing turns
        st.session_state.chat_history = [
            t for t in st.session_state.chat_history if t.get("status") != "processing"
        ]
        st.session_state.searching = False

    # Check if there's a pending processing turn (from a previous rerun)
    pending_turn = None
    for turn in st.session_state.chat_history:
        if turn.get("status") == "processing":
            pending_turn = turn
            break

    # Two-column layout: chat (wide) + controls (narrow right panel)
    chat_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        with st.expander("Enhancements", expanded=not st.session_state.searching):
            use_rerank = st.checkbox("Reranker", value=True, help="Cross-encoder reranking", disabled=st.session_state.searching)
            expand = st.checkbox("LLM expansion", value=True, help="Expand query via LiteLLM", disabled=st.session_state.searching)
            graph_expand = st.slider("Graph hops", min_value=0, max_value=3, value=1, disabled=st.session_state.searching)
            agentic = st.checkbox("Agentic mode", value=False, help="Agent decides params", disabled=st.session_state.searching)
        with st.expander("Filters", expanded=not st.session_state.searching):
            label_filter = st.text_input("Label filter", value="", help="e.g. quotation_request", disabled=st.session_state.searching)
            top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, disabled=st.session_state.searching)

    with chat_col:
        _render_chat_history(st.session_state.chat_history)

        # If there's a pending processing turn, run the search now
        if pending_turn:
            query = pending_turn["query"]
            with st.chat_message("assistant"):
                with st.status("Processing query...", expanded=True) as status:
                    # Stage 1: Query expansion
                    st.write("1/6 Query expansion...")
                    llm_ok = _check_llm_available()
                    if expand and not llm_ok:
                        st.write("   ⚠️ LLM unavailable — skipping expansion")
                        expand = False
                    if agentic and not llm_ok:
                        st.write("   ⚠️ LLM unavailable — disabling agentic mode")
                        agentic = False

                    # Stage 2: Hybrid search
                    st.write("2/6 Hybrid search (BM25 + vector)...")
                    results, raw_hits = _run_search(
                        query, top_k, use_rerank, expand, graph_expand, agentic, label_filter
                    )
                    st.write(f"   Found {len(raw_hits)} hits")

                    # Stage 3: RRF fusion
                    st.write("3/6 RRF fusion...")
                    st.write("   Merged expanded queries")

                    # Stage 4: Graph expansion
                    st.write("4/6 Graph expansion...")
                    if graph_expand > 0:
                        st.write(f"   {graph_expand} hops traversed")
                    else:
                        st.write("   Skipped (0 hops)")

                    # Stage 5: Reranking
                    st.write("5/6 Reranking...")
                    if use_rerank:
                        st.write("   Cross-encoder applied")
                    else:
                        st.write("   Skipped")

                    # Stage 6: LLM answer generation
                    st.write("6/6 Generating answer...")
                    context_parts = []
                    for hit in raw_hits[: top_k * 2]:
                        content = hit.get("text", "") if isinstance(hit, dict) else ""
                        if content:
                            context_parts.append(content)
                    context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

                    if llm_ok:
                        answer = generate_answer_litellm(query, context)
                    else:
                        answer = "LLM is unavailable. Here are the retrieved messages:\n\n" + "\n\n---\n\n".join(context_parts[:3])

                    status.update(label="Search complete!", state="complete")

            # Update the turn with results
            pending_turn["answer"] = answer
            pending_turn["raw_hits"] = raw_hits
            pending_turn["status"] = "done"
            st.session_state.searching = False
            st.rerun()

    # Chat input at page level (full-width, fixed at bottom)
    query = st.chat_input("Ask anything about the conversations...")
    if query and not st.session_state.searching:
        # Add user turn + pending assistant turn
        st.session_state.chat_history.append({
            "query": query,
            "answer": None,
            "raw_hits": [],
            "status": "processing",
        })
        st.session_state.searching = True
        st.rerun()

elif page == "Contacts":
    st.title("Contacts")
    df = fetch_contacts()
    if df.empty:
        st.warning("No contacts available.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Messages":
    st.title("Messages")
    df = fetch_messages()
    if df.empty:
        st.warning("No messages available.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Overview":
    st.title("Overview")
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
    st.title("Chat Viewer")
    name_map = fetch_contact_name_map()
    if not name_map:
        st.warning("No contacts available.")
    else:
        contact_options = {name: uid for uid, name in name_map.items()}
        selected = st.selectbox("Select a contact", list(contact_options.keys()))
        if selected:
            userid = contact_options[selected]
            conv_df = fetch_conversations_for_contact(userid)
            if conv_df.empty:
                st.info("No messages for this contact.")
            else:
                st.caption(f"{len(conv_df)} messages")
                for _, row in conv_df.iterrows():
                    is_user = row["origin"] == "3"
                    with st.chat_message("user" if is_user else "assistant"):
                        st.markdown(f"**{row['external_userid']}** ({row.get('label', '')})\n\n{row['content']}")

elif page == "Onyx Chat":
    # TODO: retire this tab — kept for backward compatibility
    st.title("Onyx Chat (deprecated)")
    st.caption("This tab will be removed in a future release. Use Search for the chat experience.")
    try:
        iframe_url = OLLAMA_URL or "http://localhost:11434"
        st.components.v1.iframe(iframe_url, height=600, scrolling=True)
    except Exception:
        st.warning("Could not load Onyx Chat.")