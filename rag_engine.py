import os
import json
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# ─── CONFIG ────────────────────────────────────────────────
CHROMA_DIR   = "gitlab_vectordb"
CHUNKS_FILE  = "gitlab_chunks_v4.json"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K        = 8
TEMPERATURE  = 0.2
BATCH_SIZE   = 500
# ────────────────────────────────────────────────────────────

def build_db_if_missing():
    """
    Build vector DB from chunks JSON if DB doesn't exist.
    Runs automatically on first deployment.
    """
    if os.path.exists(CHROMA_DIR):
        return  # Already built, skip

    if not os.path.exists(CHUNKS_FILE):
        st.error(f"{CHUNKS_FILE} not found! Cannot build database.")
        st.stop()

    st.info("⏳ Building vector database for first time... This takes 10-15 minutes. Please wait and do not close the tab.")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts     = [c["text"] for c in chunks]
    metadatas = [{"chunk_id": c["chunk_id"], "source": c["source"]} for c in chunks]

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb  = None

    for i in range(0, len(texts), BATCH_SIZE):
        bt = texts[i:i+BATCH_SIZE]
        bm = metadatas[i:i+BATCH_SIZE]
        if vectordb is None:
            vectordb = Chroma.from_texts(
                texts=bt, embedding=embedding,
                metadatas=bm, persist_directory=CHROMA_DIR
            )
        else:
            vectordb.add_texts(texts=bt, metadatas=bm)

    st.success("✅ Database built successfully! Loading GitBot...")
    st.rerun()


@st.cache_resource(show_spinner=False)
def load_models(api_key):
    """Load and cache embedding model, vector DB, LLM and retriever."""
    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb  = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding
        )
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
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
    Retrieve relevant chunks and generate answer using Gemini.
    Returns (answer, sources, error_message, not_found_flag)
    """
    try:
        # Step 1 — Retrieve using clean question only
        docs = retriever.invoke(question)

        if not docs:
            return (
                "I couldn't find relevant information in GitLab's documentation "
                "for this question. Please check handbook.gitlab.com or "
                "about.gitlab.com/direction directly.",
                [], None, True
            )

        context = "\n\n".join(doc.page_content for doc in docs)

        # Step 2 — Collect source metadata for transparency badges
        sources = list({doc.metadata.get("source", "handbook") for doc in docs})

        # Step 3 — Build prompt (history to LLM only, not retriever)
        history_section = f"Conversation history:\n{history_str}\n" if history_str else ""

        prompt_text = f"""You are GitBot, an expert assistant for GitLab's Handbook and Product Direction.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. The context DOES contain the answer — read it carefully before responding.
3. Never say you cannot find information if relevant text exists in the context.
4. Answer in full sentences with bullet points for lists.
5. If listing items (values, policies etc), list ALL of them.
6. If truly not in context, say: "I couldn't find this in GitLab's documentation. Please check handbook.gitlab.com or directions.gitlab.com directly."

{history_section}
CONTEXT (read carefully — your answer is here):
{context}

Question: {question}

Provide a thorough answer using the context above:"""

        chain  = llm | StrOutputParser()
        answer = chain.invoke(prompt_text)

        not_found = (
            "couldn't find" in answer.lower() or
            "check handbook" in answer.lower()
        )
        return answer, sources, None, not_found

    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "429" in err:
            return None, [], "Rate limit reached (15 req/min or 1,000/day free tier). Please wait a moment and try again.", False
        elif "api_key" in err.lower() or "401" in err or "403" in err:
            return None, [], "Invalid API key. Please check your GEMINI_API_KEY.", False
        elif "deadline" in err.lower() or "timeout" in err.lower():
            return None, [], "Request timed out. Please try again.", False
        else:
            return None, [], f"Something went wrong: {err}", False


def build_history_string(messages, max_history=6):
   
    recent   = messages[-(max_history * 2):]
    history  = ""
    skip_next = False

    for i, m in enumerate(recent):
        if m["role"] == "user":
            # Check if next bot message was a not-found
            if i + 1 < len(recent) and recent[i+1].get("skip_history", False):
                skip_next = True
                continue
            skip_next = False
            history += f"User: {m['content']}\n"
        else:
            if skip_next:
                skip_next = False
                continue
            history += f"GitBot: {m['content']}\n"

    return history
