import os
import streamlit as st
from dotenv import load_dotenv

# Local modules
from rag_engine import (
    build_db_if_missing,
    load_models,
    get_answer,
    build_history_string,
    TOP_K
)
from ui_components import (
    STYLES,
    HEADER_HTML,
    EMPTY_STATE_HTML,
    user_message_html,
    bot_message_html,
    error_box_html
)

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="GitBot — GitLab Assistant",
    page_icon="🦊",
    layout="centered"
)

# ─── BUILD DB IF NEEDED ────────────────────────────────────
build_db_if_missing()

# ─── LOAD ENV ──────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── INJECT CSS ────────────────────────────────────────────
st.markdown(STYLES, unsafe_allow_html=True)

# ─── API KEY CHECK ─────────────────────────────────────────
if not GEMINI_API_KEY:
    st.markdown(error_box_html(
        "GEMINI_API_KEY not found. "
        "Create a .env file with: GEMINI_API_KEY=your_key_here. "
        "Get a free key at aistudio.google.com"
    ), unsafe_allow_html=True)
    st.stop()

# ─── SESSION STATE ─────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "input_key"      not in st.session_state:
    st.session_state.input_key = 0

# ─── HEADER ────────────────────────────────────────────────
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About GitBot")
    st.markdown("""
GitBot is trained on:
- 📘 **GitLab Handbook** — culture, HR, engineering
- 🗺️ **Product Direction** — roadmap and vision

**Model:** Gemini 2.5 Flash Lite
**Vector DB:** ChromaDB
**Embeddings:** all-MiniLM-L6-v2
    """)

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages       = []
        st.session_state.question_count = 0
        st.session_state.input_key     += 1
        st.rerun()

    st.markdown(f"Questions asked: **{st.session_state.question_count}**")
    st.markdown("---")
    st.markdown("""
<div style='font-size:11px; color:#7d8590;'>
Free tier: 1,000 requests/day<br>
15 requests/minute
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ───────────────────────────────────────────
with st.spinner("Loading GitBot..."):
    retriever, llm, load_error = load_models(GEMINI_API_KEY)

if load_error:
    st.markdown(
        error_box_html(f"Failed to load models: {load_error}"),
        unsafe_allow_html=True
    )
    st.stop()

# ─── CHAT HISTORY ──────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(user_message_html(msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(
            bot_message_html(
                msg["content"],
                msg.get("sources", []),
                msg.get("chunks", TOP_K)
            ),
            unsafe_allow_html=True
        )

# ─── INPUT BOX ─────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", "")
question = st.text_input(
    "Ask anything about GitLab",
    value=prefill,
    placeholder="e.g. What is merge request?",
    label_visibility="collapsed",
    key=f"input_{st.session_state.input_key}"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("Ask →")

# ─── HANDLE SUBMIT ─────────────────────────────────────────
if ask_btn and question.strip():
    user_q = question.strip()

    # Build history (skips not-found exchanges)
    history_str = build_history_string(st.session_state.messages)

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.session_state.question_count += 1

    # Get answer
    with st.spinner("GitBot is thinking..."):
        answer, sources, error, not_found = get_answer(
            retriever, llm, user_q, history_str
        )

    if error:
        # Roll back user message so they can retry
        st.session_state.messages.pop()
        st.session_state.question_count -= 1
        st.markdown(error_box_html(error), unsafe_allow_html=True)
    else:
        st.session_state.messages.append({
            "role":         "assistant",
            "content":      answer,
            "sources":      sources,
            "chunks":       TOP_K,
            "skip_history": not_found
        })
        st.session_state.input_key += 1
        st.rerun()

# ─── EMPTY STATE ───────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)
