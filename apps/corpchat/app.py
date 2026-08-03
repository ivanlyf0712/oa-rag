#!/usr/bin/env python3
"""
CorpChat Intelligence – Streamlit App
View contacts, messages, statistics, a chat-style conversation viewer, and semantic search.
"""

import sys
import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# ── Ensure the project root (ocr/) is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.db import get_db_connection
from core.config import OLLAMA_URL, RAG_MODEL

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

# ── Page config ──
st.set_page_config(
    page_title="CorpChat Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🕵️ CorpChat Intelligence")
st.markdown("### Corporate Relationship & Chat Analytics")

# ═══════════════════════════════════════ helpers ════════════════════════════════════
@st.cache_data(ttl=30)
def fetch_contacts():
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT id, full_name, job_title, company, phone, email, userid, created_at FROM contacts ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df

@st.cache_data(ttl=30)
def fetch_messages():
    conn = get_db_connection()
    df = pd.read_sql(
        """SELECT id, msgid, open_kfid, external_userid, send_time, origin, 
                  servicer_userid, msgtype, content, label, created_at 
           FROM messages ORDER BY send_time DESC LIMIT 500""",
        conn
    )
    conn.close()
    return df

@st.cache_data(ttl=60)
def fetch_stats():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM contacts")
    total_contacts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM messages")
    total_msgs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT open_kfid) FROM messages")
    total_convos = cur.fetchone()[0]
    cur.execute("""
        SELECT label, COUNT(*) 
        FROM messages 
        GROUP BY label 
        ORDER BY COUNT(*) DESC
    """)
    label_counts = cur.fetchall()
    cur.close()
    conn.close()
    return total_contacts, total_msgs, total_convos, label_counts

# ── Chat viewer helpers ──
def get_contact_name_map():
    """Return a dict {userid: full_name} for all contacts."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT userid, full_name FROM contacts")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {row[0]: (row[1] if row[1] else row[0]) for row in rows}

def get_conversation_list(label_filter=None, search_term=None):
    """Return a list of distinct conversations with last message info."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Build WHERE clause
    conditions = []
    params = []
    if label_filter:
        conditions.append("label = %s")
        params.append(label_filter)
    if search_term:
        conditions.append("(external_userid ILIKE %s OR servicer_userid ILIKE %s)")
        params.extend([f"%{search_term}%", f"%{search_term}%"])

    where = " AND ".join(conditions) if conditions else "TRUE"

    # Get unique conversations (one row per open_kfid)
    cur.execute(f"""
        SELECT open_kfid,
               MAX(external_userid) AS external_userid,
               MAX(servicer_userid) AS servicer_userid,
               MAX(send_time) AS last_time
        FROM messages
        WHERE {where}
        GROUP BY open_kfid
        ORDER BY last_time DESC
        LIMIT 50
    """, params)

    conversations = []
    name_map = get_contact_name_map()
    for row in cur.fetchall():
        kfid, cust, agent, last_time = row
        # Get last message content
        cur.execute("SELECT content FROM messages WHERE open_kfid = %s ORDER BY send_time DESC LIMIT 1", (kfid,))
        last_msg = cur.fetchone()
        snippet = last_msg[0][:50] + "..." if last_msg and len(last_msg[0]) > 50 else (last_msg[0] if last_msg else "")

        display_name = name_map.get(cust, cust)
        conversations.append({
            "open_kfid": kfid,
            "display_name": display_name,
            "snippet": snippet,
            "last_time": last_time,
            "cust": cust,
            "agent": agent
        })

    cur.close()
    conn.close()
    return conversations

def get_messages_for_conversation(open_kfid):
    """Fetch all messages for a given conversation, ordered by time."""
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT msgid, external_userid, servicer_userid, send_time, origin, content "
        "FROM messages WHERE open_kfid = %s ORDER BY send_time ASC",
        conn, params=(open_kfid,)
    )
    conn.close()
    return df

# ════════════════════════════════ Onyx 风格搜索（由 search.py 提供）═══════════════════════
@st.cache_resource
def _load_search_index():
    """加载 search.py 构建的索引（带分块+丰富化），返回 txtai Embeddings。"""
    try:
        return load_index(DEFAULT_INDEX_PATH)
    except FileNotFoundError:
        st.warning(
            "search_index 索引不存在。请先运行 `python apps/corpchat/search.py build --force` "
            "来构建带分块和丰富化的搜索索引。"
        )
        return None

# ── 复用 search.py 的 _clean_text_from_enriched ──
from apps.corpchat.search import _clean_text_from_enriched as _search_clean_text

def search_messages_onyx(
    query: str,
    label_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 10,
    use_rerank: bool = True,
    expand: bool = True,
    graph_expand: int = 1,
):
    """
    使用 search.py 的 Searcher (Onyx §2.5-§2.8) 执行 **全链路** 搜索。

    默认启用所有增强功能 (最强大模式):
      - LLM 查询扩展 + 多查询加权 RRF 融合 (expand=True)
      - 图一跳邻居扩展 (graph_expand=1)
      - 交叉编码器重排序 (use_rerank=True)

    Searcher 已修复:
      - metadata 从 enriched text 的 "Metadata: key=value;..." 后缀中反向解析
      - label 过滤和日期过滤使用正确解析的 metadata
      - 内容文本可通过 _clean_text_from_enriched() 提取干净内容

    返回格式: [(msgid, content, send_time, external_userid, servicer_userid, label, score), ...]
    """
    embeddings = _load_search_index()
    if embeddings is None:
        return []

    query_expander = None
    if expand:
        from apps.corpchat.search import QueryExpander
        query_expander = QueryExpander()

    reranker = None
    if use_rerank:
        from apps.corpchat.search import Reranker
        reranker = Reranker()
    
    searcher = Searcher(embeddings, expander=query_expander, reranker=reranker)
    results = searcher.search(
        query=query,
        mode="hybrid",
        limit=top_k,
        expand=expand,
        graph_expand=graph_expand,
        label_filter=label_filter,
        date_from=date_from,
        date_to=date_to,
        use_rerank=use_rerank,
    )

    output = []
    for r in results:
        meta = r.get("metadata", {})
        send_time_str = meta.get("send_time")
        if send_time_str:
            try:
                send_time = datetime.fromisoformat(send_time_str)
            except (ValueError, TypeError):
                send_time = None
        else:
            send_time = None
        output.append((
            r.get("id", ""),
            r.get("text", ""),             # enriched text (含标题+内容+Metadata)
            send_time,
            meta.get("external_userid"),
            meta.get("servicer_userid"),
            meta.get("label"),
            r.get("score", 0.0),
        ))
    return output

# ── 使用 LiteLLM 生成答案（替换原 generate_answer_from_messages）──
def generate_answer_litellm(query, messages):
    """使用 LiteLLM API 生成自然语言答案。"""
    if not messages:
        return "No relevant messages found."

    # 构建上下文：取前 3 条消息的内容
    context_parts = []
    for m in messages[:3]:
        content = m[1] or ""
        if content:
            context_parts.append(content[:1000])
    if not context_parts:
        return "No message content available."

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a corporate intelligence analyst. Based on the message data below,
answer the user's question in one or two sentences. If you cannot answer, say "Insufficient data."

Question: {query}

Relevant messages:
{context}

Answer:"""

    # 调用 LiteLLM (OpenAI-compatible endpoint)
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LITELLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 128,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return "(Request timed out)"
    except Exception as e:
        return f"(Error: {e})"

# ═══════════════════════════════════════════════ Tabs ═══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Contacts", "💬 Messages", "📊 Overview", "💬 Chat Viewer", "🔍 Search", "🤖 Onyx Chat"
])

# ──────────── Tab 1: Contacts ────────────
with tab1:
    st.subheader("Business Card Contacts")
    df_contacts = fetch_contacts()
    if not df_contacts.empty:
        st.metric("Total Contacts", len(df_contacts))
        st.dataframe(df_contacts, width='stretch')
    else:
        st.info("No contacts found. Run the data generator or OCR pipeline to populate contacts.")

# ──────────── Tab 2: Messages ────────────
with tab2:
    st.subheader("WeChat Work Messages")
    df_msgs = fetch_messages()
    if not df_msgs.empty:
        st.metric("Messages (last 500)", len(df_msgs))
        origin_map = {3: "Customer", 4: "System", 5: "Agent"}
        if 'origin' in df_msgs.columns:
            df_msgs['origin'] = df_msgs['origin'].map(origin_map)
        st.dataframe(df_msgs, width='stretch')
    else:
        st.info("No messages found. Generate some conversations first.")

# ──────────── Tab 3: Overview ────────────
with tab3:
    st.subheader("Database Overview")
    total_contacts, total_msgs, total_convos, label_counts = fetch_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contacts", total_contacts)
    with col2:
        st.metric("Total Messages", total_msgs)
    with col3:
        st.metric("Unique Conversations", total_convos)

    st.subheader("Messages by Label")
    if label_counts:
        df_labels = pd.DataFrame(label_counts, columns=["Label", "Count"])
        st.bar_chart(df_labels.set_index("Label"))
    else:
        st.info("No labels found. Labels are assigned during message generation (e.g., normal, scam_crypto).")

# ── 已知标签列表（与 gen_fake_msg.py CONVERSATION_TEMPLATES 一致）──
KNOWN_LABELS = [
    "product_inquiry", "order_confirmation", "tech_support", "meeting_schedule",
    "invoice_issue", "vendor_evaluation", "software_license", "contract_renewal",
    "delivery_status", "product_demo", "payment_reminder", "quotation_request",
    "coordination", "sample_request", "training_program", "system_upgrade",
    "business_proposal", "after_service", "quality_issue", "marketing_campaign",
    "recruitment", "equipment_maintenance", "order_change", "warranty_claim",
    "annual_review", "warehouse_transfer", "partnership_discussion",
    "equipment_quote", "factory_audit",
    "old_friend_reconnect", "investment_opportunity",
]

# ──────────── Tab 4: Chat Viewer ────────────
with tab4:
    st.subheader("Conversation Viewer")

    # Initialize session state for selected conversation
    if "selected_kfid" not in st.session_state:
        st.session_state.selected_kfid = None

    col_left, col_right = st.columns([1, 2], gap="medium")

    with col_left:
        label_filter = st.selectbox(
            "Label filter",
            options=["All"] + KNOWN_LABELS,
            index=0,
        )
        search_term = st.text_input("Search participant", placeholder="name or userid")
        label_filter = None if label_filter == "All" else label_filter

        conversations = get_conversation_list(label_filter, search_term if search_term else None)

        if not conversations:
            st.info("No conversations match the filters.")
        else:
            for conv in conversations:
                kfid = conv["open_kfid"]
                name = conv["display_name"]
                snippet = conv["snippet"]
                last_time = conv["last_time"]
                # Relative time
                now = datetime.now(timezone.utc)
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                diff = now - last_time
                if diff.days == 0:
                    time_str = "today"
                elif diff.days == 1:
                    time_str = "yesterday"
                else:
                    time_str = f"{diff.days}d ago"

                if st.button(
                    f"**{name}**  \n{snippet}  \n_{time_str}_",
                    key=f"chat_{kfid}",
                    use_container_width=True
                ):
                    st.session_state.selected_kfid = kfid

    with col_right:
        if st.session_state.selected_kfid is None:
            st.info("👈 Select a conversation from the list to view the chat.")
        else:
            kfid = st.session_state.selected_kfid
            st.markdown(f"### Conversation {kfid}")
            msgs = get_messages_for_conversation(kfid)

            if msgs.empty:
                st.warning("No messages found for this conversation.")
            else:
                name_map = get_contact_name_map()
                last_date = None
                for idx, row in msgs.iterrows():
                    msg_date = row["send_time"].date()
                    if msg_date != last_date:
                        st.markdown(f"--- **{msg_date.strftime('%Y-%m-%d')}** ---")
                        last_date = msg_date

                    if row["origin"] == 3:
                        sender_name = name_map.get(row["external_userid"], row["external_userid"])
                        role = "user"
                    else:
                        sender_name = name_map.get(row["servicer_userid"], row.get("servicer_userid", "System"))
                        role = "assistant"

                    with st.chat_message(role):
                        st.markdown(f"**{sender_name}**  \n{row['content']}")
                        st.caption(row["send_time"].strftime("%H:%M"))

# ──────────── Tab 5: Onyx 风格搜索 (由 search.py 提供) ────────────
with tab5:
    st.subheader("🔍 Onyx-Style Search (全链路 — 默认最强大模式)")
    st.caption(
        "引擎: txtai hybrid (BM25 + 向量) — 默认启用: LLM 查询扩展 + RRF 融合 + 图一跳扩展 + 交叉编码器重排序 | LiteLLM: " + LITELLM_MODEL
    )

    with st.expander("📋 示例查询与预期结果", expanded=False):
        st.markdown("""
| 查询 | 预期匹配标签 | 预期匹配内容关键词 | 说明 |
|------|------------|-------------------|------|
| `詐騙` / `scam` | `old_friend_reconnect`, `investment_opportunity` | 釣魚連結, 投資方案, 邀請碼 | 兩段詐騙對話 |
| `投資方案 高回報` | `investment_opportunity` | 年化報酬率 8-12%, 債券, 藍籌股 | 羅思婷推銷投資 |
| `物流報價 方案` | `product_inquiry` | 物流系統, ¥150 萬, 報價 | 陳志明詢價 |
| `ERP timeout` | `tech_support` | timeout, 伺服器, 報表 | 張偉強報修 |
| `發票金額不符` | `invoice_issue` | ¥320,000 vs ¥350,000, 合約 | 廖珮琪核對發票 |
| `M365 E5 授權` | `software_license` | 50 個授權, ¥480,000 | 謝明宏詢價 |
| `不良率 3%` | `quality_issue` | QA, 抽檢, 零件 | 蕭國榮反應品質 |
| `Surface 電池` | `warranty_claim` | 續航力, 保固, 換貨 | 微軟保固處理 |
| `續約租金` | `contract_renewal` | 調漲 3%, 租賃合約 | 和碩續約 |
| `增加訂單 800` | `order_change` | 500→800 片, 交期 | 華碩調整數量 |
""")

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = None
    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = None

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        lbl_filter = st.selectbox(
            "Label",
            options=["All"] + KNOWN_LABELS,
            index=0,
            key="search_label"
        )
    with col_f2:
        date_from = st.text_input("Date from (YYYY-MM-DD)", placeholder="2024-01-01", key="search_date_from")
    with col_f3:
        date_to = st.text_input("Date to (YYYY-MM-DD)", placeholder="2024-12-31", key="search_date_to")
    with col_f4:
        use_rerank = st.checkbox("Rerank (cross-encoder)", value=True, key="search_rerank")

    col_adv1, col_adv2, col_adv3 = st.columns([1, 1, 2])
    with col_adv1:
        use_expand = st.checkbox("LLM query expand", value=True, key="search_expand",
                                 help="启用 LLM 查询扩展 + 多查询加权 RRF 融合")
    with col_adv2:
        graph_expand = st.number_input("Graph expand (hops)", min_value=0, max_value=3, value=1, step=1,
                                       key="search_graph_expand",
                                       help="图一跳邻居扩展数 (0=关闭)")
    with col_adv3:
        st.markdown("")  # spacer

    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        search_query = st.text_input("Search query", placeholder="e.g., 詐騙, 投資方案, 物流報價", key="search_input")
    with col_q2:
        top_k = st.slider("Results", 1, 20, 10, key="search_top_k")

    if st.button("🔍 Onyx Search", type="primary"):
        if not search_query:
            st.warning("Please enter a search query.")
        else:
            st.session_state.rag_answer = None
            lbl = None if lbl_filter == "All" else lbl_filter
            from_date = datetime.strptime(date_from, "%Y-%m-%d").isoformat() if date_from else None
            to_date = datetime.strptime(date_to, "%Y-%m-%d").isoformat() if date_to else None

            features_desc = []
            if use_expand:
                features_desc.append("LLM 扩展+RRF")
            if graph_expand > 0:
                features_desc.append(f"图{graph_expand}跳")
            if use_rerank:
                features_desc.append("重排序")
            status_text = f"Searching with {' + '.join(features_desc) if features_desc else '基础混合搜索'}..."
            with st.spinner(status_text):
                try:
                    results = search_messages_onyx(
                        search_query, lbl, from_date, to_date, top_k,
                        use_rerank=use_rerank,
                        expand=use_expand,
                        graph_expand=graph_expand,
                    )
                    st.session_state.search_results = results
                    st.session_state.search_query = search_query
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.session_state.search_results = None

    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        if results:
            st.success(f"Found {len(results)} result(s)")

            name_map = get_contact_name_map()
            df = pd.DataFrame(results, columns=[
                "Message ID", "Content", "Send Time", "Customer ID", "Agent ID", "Label", "Similarity"
            ])
            df["Customer"] = df["Customer ID"].apply(lambda x: name_map.get(x, x) if x else "")
            df["Agent"] = df["Agent ID"].apply(lambda x: name_map.get(x, x) if x else "—")
            df["Send Time"] = df["Send Time"].apply(
                lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)
            )
            df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
            df["Content"] = df["Content"].apply(lambda x: (x or "")[:100] + "..." if x and len(x) > 100 else (x or ""))

            st.dataframe(
                df[["Message ID", "Customer", "Agent", "Send Time", "Label", "Similarity", "Content"]],
                width='stretch',
                column_config={
                    "Similarity": st.column_config.TextColumn("Similarity", width="small"),
                    "Label": st.column_config.TextColumn("Label", width="small"),
                }
            )

            st.divider()
            if st.button("🤖 Generate Answer", type="secondary"):
                with st.spinner("Generating answer..."):
                    answer = generate_answer_litellm(
                        st.session_state.search_query, results
                    )
                    st.session_state.rag_answer = answer

            if st.session_state.rag_answer:
                st.markdown(f"**Answer:** {st.session_state.rag_answer}")
        else:
            st.info("No matching messages found. Make sure the index is built: `python apps/corpchat/search.py build --force`")

# ──────────── Tab 6: Onyx Chat ────────────
with tab6:
    st.subheader("🤖 Enterprise Chat RAG")

    # To find the agent ID:
    #   1. Log into Onyx at http://localhost:3000
    #   2. Navigate to Agents → Edit Agent for "Enterprise Chat RAG"
    #   3. The agent ID is the numeric portion in the URL (e.g., /agents/3 → agentId=3)
    AGENT_ID = 1  # <-- REPLACE with the actual agent ID from the Onyx Agents page

    onyx_chat_url = f"http://localhost:3000/app?agentId={AGENT_ID}"
    st.iframe(src=onyx_chat_url, height=700)

st.markdown("---")
st.caption("CorpChat Intelligence – powered by Unlimited‑OCR & RAG Pipeline")