# 🦊 GitBot — GitLab Knowledge Chatbot

> An AI-powered chatbot trained on GitLab's official Handbook and Product Direction pages. Built with RAG (Retrieval Augmented Generation), ChromaDB, LangChain, and Google Gemini API. Deployed on Streamlit Community Cloud.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [How RAG Works](#-how-rag-works)
- [Features](#-features)
- [Local Setup](#-local-setup)
- [Deployment](#-deployment)
- [Environment Variables](#-environment-variables)
- [Key Design Decisions](#-key-design-decisions)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Improvements](#-future-improvements)

---

## 📖 Project Overview

GitBot is an interactive chatbot that allows GitLab employees and aspiring employees to easily access information from GitLab's Handbook and Direction pages. Instead of manually searching through hundreds of handbook pages, users can ask natural language questions and get accurate, sourced answers instantly.

**Inspired by GitLab's "build in public" philosophy**, this project embodies transparency by:
- Showing users exactly which source (Handbook or Product Direction) each answer comes from
- Clearly stating when information is not available rather than hallucinating
- Using only GitLab's official public documentation as the knowledge base

---

## 🌐 Live Demo

> **[https://suruchi0907-gitbot.streamlit.app](https://suruchi0907-gitbot.streamlit.app)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER INTERFACE                       │
│                  (Streamlit Web App)                     │
└─────────────────────────┬───────────────────────────────┘
                          │ User Question
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   RAG PIPELINE                           │
│                                                          │
│  1. Embed Question                                       │
│     all-MiniLM-L6-v2 → 384-dim vector                   │
│                                                          │
│  2. Retrieve Relevant Chunks                             │
│     ChromaDB MMR Search → Top 8 chunks                  │
│                                                          │
│  3. Generate Answer                                      │
│     Gemini 2.5 Flash Lite + Context → Answer            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  KNOWLEDGE BASE                          │
│                                                          │
│  📘 GitLab Handbook        🗺️ Product Direction          │
│  (Git cloned .md files)   (Scraped HTML pages)          │
│                                                          │
│  Cleaned → Chunked → Embedded → ChromaDB                │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.12 | Core language |
| UI Framework | Streamlit | Latest | Web interface |
| LLM | Google Gemini 2.5 Flash Lite | API | Answer generation |
| Embeddings | all-MiniLM-L6-v2 | via HuggingFace | Text vectorization |
| Vector Database | ChromaDB | Latest | Semantic search |
| RAG Framework | LangChain | Latest | Pipeline orchestration |
| Data Scraping | BeautifulSoup + Requests | Latest | Direction page scraping |
| Data Source 1 | GitLab Handbook | Public repo | Company knowledge |
| Data Source 2 | about.gitlab.com/direction | Public website | Product roadmap |
| Deployment | Streamlit Community Cloud | - | Public hosting |

---

## 📁 Project Structure

```
gitbot/
│
├── gitlab_chatbot_ui.py      # Main Streamlit app — layout, routing, input handling
├── rag_engine.py             # Core AI logic — retrieval, generation, memory
├── ui_components.py          # All HTML/CSS — styles, chat bubbles, badges
│
├── gitlab_chunks_v4.json     # Processed knowledge base (5,066 clean chunks)
├── chunk_gitlab_data.py      # Chunking pipeline script
├── scrape_direction.py       # Direction pages web scraper
├── filter_and_rebuild.py     # Data filtering + ChromaDB rebuild script
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

### Module Responsibilities

**`gitlab_chatbot_ui.py`** — Entry point
- Streamlit page configuration
- Sidebar 
- Chat history rendering
- Input handling and session state management
- Calls `rag_engine.py` for answers and `ui_components.py` for rendering

**`rag_engine.py`** — AI Brain
- `build_db_if_missing()` — Auto-builds vector DB on first deployment
- `load_models()` — Loads and caches embedding model, ChromaDB, Gemini LLM
- `get_answer()` — Retrieves chunks and generates answers
- `build_history_string()` — Manages conversation memory, skips failed answers

**`ui_components.py`** — Visual Layer
- All CSS styles and dark theme
- HTML templates for chat bubbles, header, empty state
- Source transparency badges (📘 Handbook / 🗺️ Direction)
- Error box templates

---

## 🔄 Data Pipeline

### Step 1 — Data Collection

**GitLab Handbook** (cloned from public repo):
```bash
git clone https://gitlab.com/gitlab-com/content-sites/handbook.git
Get-ChildItem -Path . -Recurse -Filter "*.md" | Get-Content | Out-File gitlab_handbook.txt
```

**GitLab Direction Pages** (scraped):
```bash
python scrape_direction.py
# Scrapes about.gitlab.com/direction and all sub-pages
# Output: gitlab_direction.txt
```

### Step 2 — Cleaning

Raw text goes through aggressive cleaning:
- Remove HTML tags, ERB templates `<% %>`
- Strip markdown syntax (`#`, `**`, `[]()`, ` ``` `)
- Remove URLs, code blocks, table rows
- Filter lines with >50% special characters (code/config noise)
- Remove YAML config lines and file paths

### Step 3 — Noise Filtering

The handbook contains many irrelevant sections (Salesforce processes, Workday HR tools, Glean search guides). These are filtered by:
- Blocklist: sections dominated by tool names (Salesforce, Workday, Glean, Okta)
- Allowlist: sections must contain GitLab culture/product keywords to be kept

### Step 4 — Chunking

Text is split using **paragraph-based chunking** (better than fixed word-count):
- Split on double newlines first (preserves semantic units)
- Further split large paragraphs at sentence boundaries
- Chunk size: 400 words | Overlap: 80 words
- Minimum chunk size: 40 words (skip noise fragments)

### Step 5 — Quality Filtering

Each chunk is validated:
- Must be ≥40 words
- Must be ≥65% alphabetic words (filters remaining code/config)
- Must have <5% special characters

**Result: 5,066 high-quality text chunks**

### Step 6 — Embedding & Indexing

```python
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(texts, embedding, persist_directory="gitlab_vectordb")
```

---

## 🧠 How RAG Works

RAG (Retrieval Augmented Generation) combines search with language generation:

```
Question: "What is GitLab's parental leave policy?"
     │
     ▼
[Embed Question] → 384-dimensional vector
     │
     ▼
[MMR Search ChromaDB] → fetch 30 candidates, return 8 most relevant + diverse chunks
     │
     ▼
[Build Prompt]
  "Use ONLY this context to answer: [8 chunks] Question: ..."
     │
     ▼
[Gemini 2.5 Flash Lite] → Generates accurate, grounded answer
     │
     ▼
"GitLab provides 16 weeks of paid Parental Leave..."
```

## ✨ Features

### Core Features
- **Natural language Q&A** — Ask anything about GitLab in plain English
- **Conversation memory** — Remembers last 6 exchanges for follow-up questions
- **Source transparency** — Every answer shows 📘 Handbook or 🗺️ Direction badge
- **Suggested questions** — 8 pre-built questions in sidebar for quick access
- **Clear conversation** — Reset chat history with one click

### Technical Features
- **MMR Retrieval** — Maximum Marginal Relevance ensures diverse, relevant chunks
- **Smart history** — Failed/not-found answers excluded from memory to prevent poisoning
- **Auto DB builder** — Vector database rebuilds automatically on first deployment
- **Error handling** — Rate limits, invalid API keys, timeouts all show friendly messages
- **Input clearing** — Input box always clears after submission

### Guardrails
- Bot explicitly says "I couldn't find this" rather than hallucinating
- Strict prompt instructs model to use ONLY provided context
- Low temperature (0.2) ensures factual, consistent responses
- Not-found answers never stored in conversation history

---

## 💻 Local Setup

### Prerequisites
- Python 3.10 or higher
- Git
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Step 1 — Clone the repository

```bash
git clone https://github.com/Suruchi0907/gitbot.git
cd gitbot
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your free API key at [https://aistudio.google.com](https://aistudio.google.com)

### Step 4 — Build the knowledge base

The vector database is not included in the repo (too large for GitHub). Build it from the included chunks file:

```bash
python filter_and_rebuild.py
```

This will take **10-20 minutes** on first run. The database will be saved to `gitlab_vectordb/`.

Alternatively, the app will build it automatically on first launch — just be patient on first run.

### Step 5 — Run the app

```bash
streamlit run gitlab_chatbot_ui.py
```

Open your browser at `http://localhost:8501`

### Step 6 — Test it

Try these questions:
```
What is GitLab's parental leave policy?
How does GitLab onboard new employees?
Help me troubleshoot my CI/CD pipelines?
```

---

## 🚀 Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Fork this repository to your GitHub account

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **"New app"** and fill in:
   - Repository: `your-username/gitbot`
   - Branch: `master`
   - Main file path: `gitlab_chatbot_ui.py`

4. Click **"Advanced settings"** and add your secret:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_key_here"
   ```

5. Click **"Deploy!"**

> ⚠️ **Note:** On first deployment, the app will build the vector database from `gitlab_chunks_v4.json`. This takes **10-15 minutes**. Do not close the tab. Subsequent loads are instant.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key. Get free at aistudio.google.com |

### Gemini Free Tier Limits

| Model | Requests/minute | Requests/day |
|---|---|---|
| gemini-2.5-flash-lite | 15 | 1,000 |
| gemini-2.5-flash | 10 | 250 |

---

## 🎯 Key Design Decisions

### 1. RAG over Fine-tuning
Fine-tuning requires expensive GPU compute and labeled datasets. RAG is cheaper, faster to build, easier to update, and produces more grounded answers with less hallucination.

### 2. Paragraph-based Chunking
Fixed word-count chunking cuts sentences mid-way, destroying semantic meaning. Paragraph-based chunking keeps related sentences together for more coherent retrieval context.

### 3. MMR Retrieval
Standard similarity search returns the top-k most similar chunks which often contain redundant information. MMR (Maximum Marginal Relevance) fetches 30 candidates then picks 8 that are both relevant AND diverse, giving the LLM richer context.

### 4. History NOT injected into retriever
A critical design decision: conversation history is passed ONLY to the LLM prompt, never to the retriever. Injecting history into the vector search query pollutes the semantic search and returns wrong chunks.

### 5. Skip failed answers in history
When the bot says "I couldn't find this", that exchange is tagged and excluded from future conversation history. This prevents one failed answer from poisoning all subsequent queries.

### 6. Section-level noise filtering
The GitLab handbook contains many pages about internal tools (Salesforce, Workday, Glean) irrelevant to a knowledge chatbot. Filtering at the section level before chunking dramatically improves retrieval quality.

---

## ⚙️ RAG Configuration

```python
# rag_engine.py
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Free tier: 1000 req/day
TOP_K        = 8                         # Chunks retrieved per query
TEMPERATURE  = 0.2                       # Low = factual, consistent
CHUNK_SIZE   = 800                       # Words per chunk
OVERLAP      = 150                        # Words overlap between chunks
FETCH_K      = 30                        # MMR candidate pool size
LAMBDA_MULT  = 0.7                       # MMR relevance vs diversity balance
MAX_HISTORY  = 6                         # Conversation exchanges remembered
```

---

## 🐛 Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| Bot returning wrong chunks     | Training data dominated by sales/IT content | Section-level keyword filtering before chunking |
| Bot ignoring retrieved context | Prompt not explicit enough                  | Added "The context DOES contain the answer — read carefully" |
| History poisoning queries      | Failed answers stored in memory             | Tag not-found answers, skip them in history builder |
| History polluting retrieval    | History injected into vector search         | Pass history only to LLM prompt, never to retriever |
| Vector DB too large for GitHub | chroma.sqlite3 = 126MB                      | Excluded from repo, rebuilt from chunks JSON on deployment |
| Gemini model deprecated        | 1.5-flash retired                           | Switched to gemini-2.5-flash-lite |
| Rate limits hitting frequently | Only 250 req/day on flash                   | Switched to flash-lite (1,000 req/day) |
| Input box not clearing         | Streamlit widget key caching                | Increment input_key after each submission |

---

## 🔮 Future Improvements

- **Hybrid Search** — Combine keyword (BM25) + semantic search for better recall
- **Re-ranking** — Add a cross-encoder re-ranker to improve chunk precision
- **Better embeddings** — Use `text-embedding-3-small` (OpenAI) for higher quality vectors
- **Metadata filtering** — Allow users to search only Handbook OR only Direction
- **Authentication** — Restrict access to verified GitLab employees via Google OAuth
- **Feedback system** — Thumbs up/down on answers to improve over time
- **Answer caching** — Cache common questions to reduce API calls
- **Streaming responses** — Stream Gemini output token by token for better UX
- **Multi-language support** — Support questions in languages other than English

---

## 📄 License

This project uses GitLab's publicly available Handbook and Direction documentation.

---

## 👩‍💻 Author

Built by **Suruchi** as part of a Generative AI project assignment.

- GitHub: [@Suruchi0907](https://github.com/Suruchi0907)
- Project inspired by GitLab's "build in public" philosophy
