import json
import re
import os

# ─── CONFIG ──────────────────────────────────────────────
INPUT_FILE  = "gitlab_final_training.txt"   
OUTPUT_FILE = "gitlab_chunks.json"

CHUNK_SIZE    = 800    
CHUNK_OVERLAP = 150    
MIN_CHUNK     = 50     
# ─────────────────────────────────────────────────────────


def clean_text(text):
    """Basic cleanup — remove excessive whitespace and junk lines."""
    # Remove markdown artifacts
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)              # images
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)   # links → keep text
    text = re.sub(r'#{1,6}\s*', '', text)                    # headings
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)           # inline code
    text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', text)   # bold/italic
    text = re.sub(r'\|.*?\|', '', text)                      # tables
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)   # dividers
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE) # blockquotes
    text = re.sub(r'http\S+', '', text)                      # URLs
    text = re.sub(r'<%.*?%>', '', text, flags=re.DOTALL)     # ERB tags
    text = re.sub(r'[ \t]+', ' ', text)                      # collapse spaces
    text = re.sub(r'\n{3,}', '\n\n', text)                   # collapse blank lines
    text = text.replace("GITBOT_SOURCE_DIRECTION", "")
    return text.strip()


def split_into_chunks(text, chunk_size=800, overlap=150):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_words) >= MIN_CHUNK:
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def detect_source(text):
    if "GITBOT_SOURCE_DIRECTION" in text:
        return "direction"
    lower = text.lower()
    hits = sum(1 for kw in DIRECTION_KEYWORDS if kw in lower)
    if hits >= 2:
        return "direction"
    return "handbook"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: '{INPUT_FILE}' not found.")
        print("Make sure you are running this from the folder containing your merged file.")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    print("Cleaning text...")
    cleaned = clean_text(raw)

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_into_chunks(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"Total chunks created: {len(chunks)}")

    # Build JSON output with metadata
    output = []
    for i, chunk in enumerate(chunks):
        output.append({
            "chunk_id": i,
            "source": detect_source(chunk),
            "word_count": len(chunk.split()),
            "text": chunk
        })

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Stats
    word_counts = [c["word_count"] for c in output]
    print("\n✅ Chunking complete!")
    print(f"   Total chunks    : {len(output)}")
    print(f"   Avg chunk size  : {sum(word_counts) // len(word_counts)} words")
    print(f"   Min chunk size  : {min(word_counts)} words")
    print(f"   Max chunk size  : {max(word_counts)} words")
    print(f"   Output saved to : {OUTPUT_FILE}")
    print("\nNext step: Load gitlab_chunks.json into your vector DB (FAISS, Chroma, Pinecone etc.)")


if __name__ == "__main__":
    main()
