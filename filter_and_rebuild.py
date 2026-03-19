import sys
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "langchain-community", "langchain-huggingface"], capture_output=True)

import json
import re
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

INPUT_FILE  = "gitlab_final_training.txt"
CLEAN_FILE  = "gitlab_filtered.txt"
CHUNKS_FILE = "gitlab_chunks_v4.json"
CHROMA_DIR  = "gitlab_vectordb"
CHUNK_SIZE  = 400
OVERLAP     = 80
MIN_WORDS   = 40

# ─── TOPICS TO REMOVE ──────────────────────────────────────
# These are tool/ops sections irrelevant to a GitLab knowledge bot
BLOCKLIST_SECTION = [
    "salesforce", "workday", "glean", "golinks", "okta",
    "netsuite", "quickbooks", "coupa", "zip ", "expensify",
    "amer swag", "expense report", "it helpdesk", "it help desk",
    "procurement process", "vendor management", "accounts payable",
    "accounts receivable", "lead database", "crm ", "deal desk",
    "quote to cash", "order management", "renewal process",
    "licensing and renewal", "amer licensing",
]

# These are keywords that MUST appear in a section for it to be kept
# (we keep sections about GitLab product, culture, engineering, HR)
ALLOWLIST_TOPICS = [
    "gitlab", "handbook", "remote", "values", "credit",
    "collaboration", "iteration", "transparency", "efficiency",
    "diversity", "results", "merge request", "pipeline", "ci/cd",
    "devsecops", "onboard", "parental", "performance", "review",
    "product", "direction", "roadmap", "vision", "engineering",
    "security", "deploy", "monitor", "plan", "create", "verify",
    "culture", "hiring", "benefit", "compensation", "leave",
    "all-remote", "async", "communication", "team member",
]


def is_blocked_section(text_block):
    """Return True if this block is dominated by noise topics."""
    lower = text_block.lower()
    # Count blocklist hits
    block_hits = sum(1 for kw in BLOCKLIST_SECTION if kw in lower)
    # Count allowlist hits
    allow_hits = sum(1 for kw in ALLOWLIST_TOPICS if kw in lower)
    # Block if blocklist dominates
    if block_hits >= 3 and allow_hits < 2:
        return True
    if block_hits >= 5:
        return True
    return False


def clean_text(text):
    """Clean markdown, HTML and noise from text."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'<%.*?%>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\*{1,3}([^\*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-|: ]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^\s*[-*+•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_junk_line(line):
    """Return True if line is code, config or noise."""
    l = line.strip()
    if not l or len(l) < 15:
        return True
    alpha = sum(c.isalpha() for c in l)
    if alpha / max(len(l), 1) < 0.45:
        return True
    if l.startswith(('$', '{', '}', '[', ']', '<', '>', '|', '#!', '//', '/*')):
        return True
    if re.match(r'^[a-z_]+:\s+\S', l) and len(l.split()) < 5:
        return True
    if sum(c in '{}[]<>|\\=+*@%^&;' for c in l) > 4:
        return True
    return False


def chunk_text(text):
    """Split into semantic paragraph-based chunks."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) < MIN_WORDS:
            continue
        if len(words) <= CHUNK_SIZE:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current, cur_words = [], 0
            for sent in sentences:
                sw = len(sent.split())
                if cur_words + sw > CHUNK_SIZE and current:
                    chunks.append(' '.join(current))
                    current = current[-2:] if len(current) >= 2 else []
                    cur_words = sum(len(s.split()) for s in current)
                current.append(sent)
                cur_words += sw
            if current and cur_words >= MIN_WORDS:
                chunks.append(' '.join(current))
    return chunks


def is_good_chunk(chunk):
    """Return True if chunk is quality prose."""
    words = chunk.split()
    if len(words) < MIN_WORDS:
        return False
    alpha_words = sum(1 for w in words if sum(c.isalpha() for c in w) > len(w) * 0.5)
    if alpha_words / len(words) < 0.65:
        return False
    if sum(c in '{}[]<>|\\=+*@%^&;`' for c in chunk) / len(chunk) > 0.05:
        return False
    return True


# ─── STEP 1: READ ──────────────────────────────────────────
print("=" * 60)
print("GitLab Data Filter + Vector DB Rebuild")
print("=" * 60)

print("\n[1/6] Reading file...")
with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()
print(f"      Size: {len(raw):,} chars")

# ─── STEP 2: SPLIT INTO SECTIONS ───────────────────────────
print("\n[2/6] Splitting into sections and filtering noise...")
# Split on double newlines to get natural sections
sections     = raw.split('\n\n')
kept         = []
removed      = 0

for section in sections:
    if is_blocked_section(section):
        removed += 1
    else:
        kept.append(section)

filtered_text = '\n\n'.join(kept)
print(f"      Sections kept   : {len(kept):,}")
print(f"      Sections removed: {removed:,}")
print(f"      Size after filter: {len(filtered_text):,} chars "
      f"({100*len(filtered_text)//len(raw)}% of original)")

# ─── STEP 3: CLEAN ─────────────────────────────────────────
print("\n[3/6] Cleaning text...")
lines      = filtered_text.splitlines()
clean_lines = [l for l in lines if not is_junk_line(l)]
cleaned    = clean_text('\n'.join(clean_lines))
print(f"      After cleaning: {len(cleaned):,} chars")

# Save filtered file for inspection
with open(CLEAN_FILE, "w", encoding="utf-8") as f:
    f.write(cleaned)
print(f"      Saved filtered text to: {CLEAN_FILE}")

# ─── STEP 4: CHUNK ─────────────────────────────────────────
print("\n[4/6] Chunking...")
raw_chunks  = chunk_text(cleaned)
good_chunks = [c for c in raw_chunks if is_good_chunk(c)]
print(f"      Raw chunks : {len(raw_chunks):,}")
print(f"      Good chunks: {len(good_chunks):,}")

output = []
for i, chunk in enumerate(good_chunks):
    source = "direction" if "=== SOURCE:" in chunk else "handbook"
    output.append({
        "chunk_id":   i,
        "source":     source,
        "word_count": len(chunk.split()),
        "text":       chunk
    })

with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"      Saved to: {CHUNKS_FILE}")

# ─── STEP 5: VERIFY KEY TOPICS IN CHUNKS ───────────────────
print("\n[5/6] Verifying key topics exist in chunks...")
all_chunk_text = " ".join(good_chunks).lower()
check_keywords = [
    "core values", "all-remote", "parental leave",
    "CREDIT", "collaboration", "iteration", "transparency"
]
for kw in check_keywords:
    count  = all_chunk_text.count(kw.lower())
    status = "✅" if count > 0 else "❌"
    print(f"      {status} '{kw}' — {count} times in chunks")

# ─── STEP 6: REBUILD CHROMADB ──────────────────────────────
print(f"\n[6/6] Rebuilding ChromaDB...")
if os.path.exists(CHROMA_DIR):
    print("      Deleting old database...")
    shutil.rmtree(CHROMA_DIR)

texts     = [c["text"] for c in output]
metadatas = [{"chunk_id": c["chunk_id"], "source": c["source"]} for c in output]

print("      Loading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BATCH_SIZE = 500
vectordb   = None
total      = len(texts)

for i in range(0, total, BATCH_SIZE):
    bt  = texts[i:i+BATCH_SIZE]
    bm  = metadatas[i:i+BATCH_SIZE]
    pct = min(100, int((i + BATCH_SIZE) / total * 100))
    print(f"      Batch {i//BATCH_SIZE+1}/{(total-1)//BATCH_SIZE+1} ({pct}%)...")
    if vectordb is None:
        vectordb = Chroma.from_texts(
            texts=bt, embedding=embedding,
            metadatas=bm, persist_directory=CHROMA_DIR
        )
    else:
        vectordb.add_texts(texts=bt, metadatas=bm)

print(f"\n{'='*60}")
print(f"DONE!")
print(f"  Clean chunks : {total:,}")
print(f"  Vector DB    : {CHROMA_DIR}/")
print(f"  Filtered text: {CLEAN_FILE}")
print(f"\nNow run: streamlit run gitlab_chatbot_ui.py")
print(f"{'='*60}")