"""
GitBot — Web UI with Gemini API, Error Handling & Source Transparency
======================================================================
Run with:  streamlit run gitlab_chatbot_ui.py

Setup:
    1. pip install streamlit langchain-chroma langchain-huggingface
                   langchain-google-genai google-generativeai python-dotenv
    2. Create a .env file in the same folder with:
       GEMINI_API_KEY=your_key_here
    3. streamlit run gitlab_chatbot_ui.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


def build_db_if_missing():
    """Build vector DB from chunks file if DB doesn't exist yet."""
    if os.path.exists("gitlab_vectordb"):
        return  # Already built, skip

    if not os.path.exists("gitlab_chunks_v4.json"):
        st.error("gitlab_chunks_v4.json not found! Cannot build database.")
        st.stop()

    st.info("Building vector database for first time... This takes 10-15 minutes. Please wait.")

    with open("gitlab_chunks_v4.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts     = [c["text"] for c in chunks]
    metadatas = [{"chunk_id": c["chunk_id"], "source": c["source"]} for c in chunks]

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    BATCH_SIZE = 500
    vectordb   = None
    for i in range(0, len(texts), BATCH_SIZE):
        bt = texts[i:i+BATCH_SIZE]
        bm = metadatas[i:i+BATCH_SIZE]
        if vectordb is None:
            vectordb = Chroma.from_texts(
                texts=bt, embedding=embedding,
                metadatas=bm, persist_directory="gitlab_vectordb"
            )
        else:
            vectordb.add_texts(texts=bt, metadatas=bm)

    st.success("Database built! Refreshing...")
    st.rerun()

# Build DB if needed (runs on first deployment)
build_db_if_missing()

# ─── CONFIG ────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_DIR     = "gitlab_vectordb"
GEMINI_MODEL   = "gemini-2.5-flash-lite"
TOP_K          = 8
TEMPERATURE    = 0.2
MAX_HISTORY    = 6
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GitBot — GitLab Assistant",
    page_icon="🦊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
#MainMenu, footer, header { visibility: hidden; }

.gitbot-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}
.gitbot-logo {
    width: 42px; height: 42px; background: #fc6d26;
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 22px; flex-shrink: 0;
}
.gitbot-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px; font-weight: 500; color: #e6edf3; margin: 0;
}
.gitbot-subtitle { font-size: 13px; color: #7d8590; margin: 2px 0 0; }

.chat-message {
    display: flex; gap: 12px;
    margin-bottom: 1.5rem; align-items: flex-start;
}
.chat-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace; font-weight: 500;
}
.avatar-user { background: #1f6feb; color: #e6edf3; }
.avatar-bot  { background: #fc6d26; color: #fff; }

.chat-bubble {
    flex: 1; padding: 12px 16px; border-radius: 0 12px 12px 12px;
    font-size: 14px; line-height: 1.7; color: #e6edf3;
    background: #161b22; border: 1px solid #21262d;
}
.bubble-user { border-radius: 12px 0 12px 12px; }
.bubble-bot ul  { margin: 8px 0 0 18px; padding: 0; }
.bubble-bot li  { margin-bottom: 4px; }
.bubble-bot strong { color: #fc6d26; }

.source-row {
    display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;
}
.source-badge {
    font-size: 11px; font-family: 'IBM Plex Mono', monospace;
    padding: 3px 10px; border-radius: 20px; border: 1px solid #30363d;
}
.badge-handbook  { background: #0d2137; color: #58a6ff; border-color: #1f6feb; }
.badge-direction { background: #1a1200; color: #e3b341; border-color: #9e6a03; }
.badge-chunks    { background: #161b22; color: #7d8590; border-color: #30363d; }

.error-box {
    background: #2d1b1b; border: 1px solid #f85149;
    border-radius: 10px; padding: 12px 16px;
    font-size: 13px; color: #f85149; margin-bottom: 1rem;
}

.stTextInput > div > div > input {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 10px !important; color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important; padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #fc6d26 !important;
    box-shadow: 0 0 0 3px rgba(252,109,38,0.15) !important;
}
.stButton > button {
    background: #fc6d26 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 14px !important;
    padding: 8px 20px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
section[data-testid="stSidebar"] {
    background: #161b22 !important; border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# ─── API KEY CHECK ─────────────────────────────────────────
if not GEMINI_API_KEY:
    st.markdown("""
<div class="error-box">
    ⚠️ <strong>GEMINI_API_KEY not found.</strong><br><br>
    Create a <code>.env</code> file in the same folder as this script:<br><br>
    <code>GEMINI_API_KEY=your_key_here</code><br><br>
    Get a free key at aistudio.google.com
</div>
""", unsafe_allow_html=True)
    st.stop()


# ─── LOAD MODELS (cached — runs only once) ─────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb  = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding
        )
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE
        )
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": 30, "lambda_mult": 0.7}
        )
        return retriever, llm, None
    except Exception as e:
        return None, None, str(e)


def get_answer(retriever, llm, question, history_str):
    """
    Retrieval uses ONLY the clean question — no history injected.
    History is passed only to the LLM prompt for conversational context.
    Returns (answer, sources, error_message)
    """
    try:
        # Step 1 — retrieve relevant chunks using clean question only
        docs = retriever.invoke(question)

        if not docs:
            return (
                "I couldn't find relevant information in GitLab's documentation "
                "for this question. Please check handbook.gitlab.com or "
                "about.gitlab.com/direction directly.",
                [],
                None
            )

        context = "\n\n".join(doc.page_content for doc in docs)

        # Step 2 — collect unique sources for transparency badges
        sources = list({doc.metadata.get("source", "handbook") for doc in docs})

        # Step 3 — build prompt with history for LLM only
        history_section = f"Conversation history:\n{history_str}\n" if history_str else ""

        prompt_text = f"""You are GitBot, an expert assistant trained on GitLab's official
Handbook and Product Direction documentation.

Use ONLY the context below to answer. Do not guess or make things up.
- Answer in full sentences, never one-word answers.
- If listing items (values, policies, stages etc), list ALL of them.
- Use bullet points for multi-part answers.
- If the answer is not in the context say exactly:
  "I couldn't find this in GitLab's documentation. Please check handbook.gitlab.com directly."

{history_section}--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
Answer:"""

        chain  = llm | StrOutputParser()
        answer = chain.invoke(prompt_text)
        return answer, sources, None

    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "429" in err:
            return None, [], (
                "Rate limit reached (15 req/min or 1,500/day free tier). "
                "Please wait a moment and try again."
            )
        elif "api_key" in err.lower() or "401" in err or "403" in err:
            return None, [], (
                "Invalid API key. Please check your GEMINI_API_KEY in the .env file."
            )
        elif "deadline" in err.lower() or "timeout" in err.lower():
            return None, [], "Request timed out. Please try again."
        else:
            return None, [], f"Something went wrong: {err}"


def build_source_badges(sources, chunk_count):
    """Render source transparency badges below each bot answer."""
    badges = ""
    if "handbook" in sources:
        badges += '<span class="source-badge badge-handbook">📘 GitLab Handbook</span>'
    if "direction" in sources:
        badges += '<span class="source-badge badge-direction">🗺️ Product Direction</span>'
    badges += f'<span class="source-badge badge-chunks">🔍 {chunk_count} chunks retrieved</span>'
    return f'<div class="source-row">{badges}</div>'


# ─── SESSION STATE ─────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "input_key"      not in st.session_state:
    st.session_state.input_key = 0


# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="gitbot-header">
    <div class="gitbot-logo">🦊</div>
    <div>
        <div class="gitbot-title">GitBot</div>
        <div class="gitbot-subtitle">
            Trained on GitLab Handbook &amp; Product Direction
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About GitBot")
    st.markdown("""
GitBot is trained on:
- 📘 **GitLab Handbook** — culture, HR, engineering
- 🗺️ **Product Direction** — roadmap and vision

**Model:** Gemini 2.5 Flash - Lite
**Vector DB:** ChromaDB
**Chunks:** 5,066
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
Free tier limits:<br>
1,500 requests/day<br>
15 requests/minute
</div>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ───────────────────────────────────────────
with st.spinner("Loading GitBot..."):
    retriever, llm, load_error = load_models()

if load_error:
    st.markdown(
        f'<div class="error-box">❌ Failed to load: {load_error}</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ─── CHAT HISTORY ──────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="chat-message">
    <div class="chat-avatar avatar-user">U</div>
    <div class="chat-bubble bubble-user">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)
    else:
        badges = build_source_badges(
            msg.get("sources", []),
            msg.get("chunks", TOP_K)
        )
        st.markdown(f"""
<div class="chat-message">
    <div class="chat-avatar avatar-bot">G</div>
    <div class="chat-bubble bubble-bot">
        {msg["content"]}
        {badges}
    </div>
</div>""", unsafe_allow_html=True)


# ─── INPUT ─────────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", "")
question = st.text_input(
    "Ask anything about GitLab",
    value=prefill,
    placeholder="e.g. What are GitLab's CREDIT values?",
    label_visibility="collapsed",
    key=f"input_{st.session_state.input_key}"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("Ask →")


# ─── HANDLE SUBMIT ─────────────────────────────────────────
if ask_btn and question.strip():
    user_q = question.strip()

    # Build history from last MAX_HISTORY exchanges
    history_msgs = st.session_state.messages[-(MAX_HISTORY * 2):]
    history_str  = ""
    for m in history_msgs:
        role = "User" if m["role"] == "user" else "GitBot"
        history_str += f"{role}: {m['content']}\n"

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.session_state.question_count += 1

    # Get answer
    with st.spinner("GitBot is thinking..."):
        answer, sources, error = get_answer(retriever, llm, user_q, history_str)

    if error:
        # Roll back user message so they can retry cleanly
        st.session_state.messages.pop()
        st.session_state.question_count -= 1
        st.markdown(
            f'<div class="error-box">❌ {error}</div>',
            unsafe_allow_html=True
        )
    else:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "chunks":  TOP_K
        })
        st.session_state.input_key += 1
        st.rerun()


# ─── EMPTY STATE ───────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div style='text-align:center; padding:3rem 0; color:#7d8590;'>
    <div style='font-size:48px; margin-bottom:16px;'>🦊</div>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:15px;
                color:#e6edf3; margin-bottom:8px;'>
        Ask me anything about GitLab
    </div>
    <div style='font-size:13px;'>
        Try a suggested question from the sidebar, or type your own below.
    </div>
</div>
""", unsafe_allow_html=True)