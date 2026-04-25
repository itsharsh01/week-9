# =============================================================================
# NCERT Mini Project - RAG-based Q&A System
# =============================================================================
# Dependencies (all pure-Python, no native DLLs):
#   python -m pip install pdfplumber rank_bm25 transformers nltk requests
# =============================================================================

import os
import re
import pdfplumber
import nltk
from nltk.corpus import stopwords
from collections import Counter
import requests
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PDF_FILE_PATH = "ncert-9.pdf"          # Update path as needed
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable

nltk.download("stopwords", quiet=True)

# ------------------------------------------------------------------
# Step 1: PDF Text Extraction
# ------------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file, starting from page 12.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF, or an empty string if an error occurs.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as doc:
            for page in doc.pages[12:]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""

# ------------------------------------------------------------------
# Step 2: Basic Keyword Analysis
# ------------------------------------------------------------------

def analyse_keywords(extracted_text):
    """Prints top frequent words and counts for various keyword groups."""
    stop_words = set(stopwords.words("english"))
    words = re.findall(r"\b\w+\b", extracted_text.lower())
    filtered_words = [w for w in words if w not in stop_words]
    filtered_counter = Counter(filtered_words)

    print("Top 30 most common words after removing stop words:")
    print(filtered_counter.most_common(30))

    def check_keywords(keywords):
        for word in keywords:
            count = filtered_words.count(word.lower())
            print(f"{word} -> {count}")

    example_keywords = [
        "example", "illustration", "illustrate", "illustrated",
        "worked example", "case", "case study", "sample",
    ]

    question_keywords = [
        "exercise", "question", "questions", "problem",
        "problems", "q.", "q", "exercises", "solve",
        "evaluate", "find", "calculate", "explain",
        "define", "derive", "state",
    ]

    concept_keywords = [
        "is defined as", "refers to", "means", "definition",
        "called", "known as", "states that", "can be defined as",
    ]

    law_keywords = [
        "law", "theorem", "principle", "rule",
        "newton", "archimedes", "ohm", "boyle",
    ]

    formula_keywords = [
        "=", "+", "-", "*", "/", "^",
        "equation", "formula", "calculate",
        "where", "given", "thus",
    ]

    print("\n--- Example Keywords ---")
    check_keywords(example_keywords)
    print("\n--- Question Keywords ---")
    check_keywords(question_keywords)
    print("\n--- Concept Keywords ---")
    check_keywords(concept_keywords)
    print("\n--- Law Keywords ---")
    check_keywords(law_keywords)
    print("\n--- Formula Keywords ---")
    check_keywords(formula_keywords)

# ------------------------------------------------------------------
# Step 3: Text Cleaning & Block Splitting
# ------------------------------------------------------------------

def clean_text(text):
    """Removes extra newlines and normalises whitespace."""
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_boundary(line):
    """Returns True if a line marks a logical section boundary."""
    line = line.strip()
    if not line:
        return False
    lower = line.lower()
    return (
        lower.startswith(("example", "illustration"))
        or lower.startswith(("exercise", "questions"))
        or lower.startswith(("q.", "q "))
        or line.startswith(tuple(str(i) + "." for i in range(1, 20)))
        or (len(line) < 60 and line.isupper())
    )


def split_into_blocks(text):
    """Splits raw text into logical blocks using boundary detection."""
    lines = text.split("\n")
    blocks = []
    current_block = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_boundary(line):
            if current_block:
                blocks.append(current_block.strip())
                current_block = ""
            blocks.append(line)
        else:
            current_block = (current_block + " " + line).strip() if current_block else line

    if current_block:
        blocks.append(current_block.strip())

    return blocks

# ------------------------------------------------------------------
# Step 4: Block Classification
# ------------------------------------------------------------------

def classify_block(text):
    """Assigns a semantic type to a text block."""
    t = text.lower()

    if text.strip().startswith(tuple(str(i) + "." for i in range(1, 20))):
        return "question"
    if any(k in t for k in ["question", "questions", "q"]):
        return "question"
    if any(k in t for k in ["example", "illustration", "case", "sample"]):
        return "example"
    if "=" in text:
        return "formula"

    return "concept"


def build_structured_blocks(text):
    """Returns a list of dicts with 'text' and 'type' for each block."""
    blocks = split_into_blocks(text)
    return [{"text": b, "type": classify_block(b)} for b in blocks]

# ------------------------------------------------------------------
# Step 5: Token-based Chunking
# ------------------------------------------------------------------

def chunk_blocks(structured_blocks, chunk_size=25, overlap=10):
    """
    Splits structured blocks into token-bounded chunks with overlap.

    Args:
        structured_blocks (list): Output of build_structured_blocks().
        chunk_size (int): Maximum tokens per chunk.
        overlap (int): Token overlap between consecutive chunks.

    Returns:
        list: List of chunk dicts with 'text' and 'type'.
    """
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    chunks = []

    for block in structured_blocks:
        text = block["text"]
        block_type = block["type"]
        tokens = tokenizer.encode(text)

        if len(tokens) <= chunk_size:
            chunks.append({"text": text, "type": block_type})
            continue

        start = 0
        while start < len(tokens):
            chunk_tokens = tokens[start : start + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append({"text": chunk_text, "type": block_type})
            start += chunk_size - overlap

    return chunks

# ------------------------------------------------------------------
# Step 6: BM25 Retrieval
# ------------------------------------------------------------------

def build_bm25_index(chunks):
    """Builds and returns a BM25 index over the provided chunks."""
    corpus = [chunk["text"] for chunk in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def retrieve(query, bm25, chunks, top_k=5):
    """
    Retrieves the top-k most relevant chunks for a query.

    Args:
        query (str): The user question.
        bm25 (BM25Okapi): Pre-built BM25 index.
        chunks (list): The original chunk list aligned with the index.
        top_k (int): Number of results to return.

    Returns:
        list: Top-k chunk dicts with an added 'score' field.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return [
        {"text": chunks[i]["text"], "type": chunks[i]["type"], "score": scores[i]}
        for i in ranked_indices[:top_k]
    ]


def is_low_confidence(retrieved_chunks, threshold=0.5):
    """Returns True if the top-3 retrieved chunks all have scores below threshold."""
    top_chunks = retrieved_chunks[:3]
    low_count = sum(1 for c in top_chunks if c["score"] < threshold)
    return low_count >= 3

# ------------------------------------------------------------------
# Step 7: Prompt Building & LLM Answer Generation
# ------------------------------------------------------------------

def build_prompt(query, retrieved_chunks, low_confidence=False):
    """Builds a grounded prompt for the LLM."""
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Chunk {i+1} | {chunk['type']}]\n{chunk['text']}\n\n"

    instruction = (
        "The retrieved context may NOT be reliable.\n\n"
        'If the answer is not clearly present, say: "I don\'t know based on the provided content."'
        if low_confidence
        else "Answer ONLY from the given context.\n\n"
        'If the answer is not present, say: "I don\'t know based on the provided content."'
    )

    return f"""You are a science tutor.

{instruction}

Context:
{context}
Question:
{query}

Answer:
"""


def generate_answer(prompt, api_key):
    """
    Calls the Gemini REST API directly via requests and returns the generated text.

    Args:
        prompt (str): The full prompt string.
        api_key (str): Gemini API key.

    Returns:
        str: The model's response text.
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def answer(question, bm25, chunks, api_key):
    """
    Full RAG pipeline: retrieve → confidence check → prompt → generate.

    Args:
        question (str): The user's question.
        bm25 (BM25Okapi): Pre-built retrieval index.
        chunks (list): Token chunks aligned with the index.
        api_key (str): Gemini API key.

    Returns:
        dict: Contains 'question', 'answer', 'chunks', and 'low_confidence'.
    """
    retrieved_chunks = retrieve(question, bm25, chunks, top_k=5)
    low_conf = is_low_confidence(retrieved_chunks)
    prompt = build_prompt(question, retrieved_chunks, low_confidence=low_conf)
    answer_text = generate_answer(prompt, api_key)

    return {
        "question": question,
        "answer": answer_text,
        "chunks": retrieved_chunks,
        "low_confidence": low_conf,
    }

# ------------------------------------------------------------------
# Step 8: CBSE Q&A Evaluation
# ------------------------------------------------------------------

CBSE_QUESTIONS = [
    # Chapter 1 – Matter in Our Surroundings
    "What are the characteristics of particles of matter?",
    # Chapter 2 – Is Matter Around Us Pure
    "What is the difference between a mixture and a compound?",
    # Chapter 3 – Atoms and Molecules
    "State the law of conservation of mass.",
    # Chapter 8 – Motion
    "What is the difference between speed and velocity?",
    # Chapter 9 – Force and Laws of Motion
    "State Newton's second law of motion.",
    # Chapter 10 – Gravitation
    "What is the universal law of gravitation?",
    # Chapter 11 – Work and Energy
    "What is kinetic energy? Give its formula.",
    # Chapter 12 – Sound
    "What is the difference between loudness and intensity of sound?",
    # Chapter 13 – Why Do We Fall Ill
    "What are the differences between infectious and non-infectious diseases?",
    # Chapter 14 – Natural Resources
    "What is the role of the atmosphere in climate control?",
]


def run_cbse_questions(bm25, chunks, api_key):
    """
    Runs all 10 CBSE science questions through the RAG pipeline and
    prints each question, its confidence level, and the generated answer.

    Args:
        bm25 (BM25Okapi): Pre-built retrieval index.
        chunks (list): Token chunks aligned with the index.
        api_key (str): Gemini API key.
    """
    print("\n" + "=" * 70)
    print("  CBSE Class 9 Science — RAG Q&A Evaluation")
    print("=" * 70)

    for idx, question in enumerate(CBSE_QUESTIONS, start=1):
        print(f"\n{'─' * 70}")
        print(f"Q{idx}: {question}")
        print("─" * 70)

        result = answer(question, bm25, chunks, api_key)

        if result["low_confidence"]:
            print("[⚠]  Low retrieval confidence — answer may be unreliable.")

        print(f"ANSWER:\n{result['answer'].strip()}")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # 1. Extract text
    extracted_text = extract_text_from_pdf(PDF_FILE_PATH)
    if not extracted_text:
        print(f"Failed to extract text from {PDF_FILE_PATH}")
        return

    print(extracted_text[:500])
    print(f"\nTotal characters extracted: {len(extracted_text)}")

    # 2. (Optional) keyword analysis
    analyse_keywords(extracted_text)

    # 3. Build structured blocks & chunks
    structured_blocks = build_structured_blocks(extracted_text)

    print("\n--- Sample Structured Blocks (10-50) ---")
    for b in structured_blocks[10:50]:
        print("\nTYPE:", b["type"])
        print(b["text"])

    tokenized_chunks = chunk_blocks(structured_blocks)

    print("\n--- Sample Chunks (first 5) ---")
    for c in tokenized_chunks[:5]:
        print("\nTYPE:", c["type"])
        print(c["text"])

    # 4. Build retrieval index
    bm25 = build_bm25_index(tokenized_chunks)

    # 5. Validate API key
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before running: $env:GEMINI_API_KEY='your_key'"
        )

    # 6. Run all 10 CBSE questions through the RAG pipeline
    run_cbse_questions(bm25, tokenized_chunks, GEMINI_API_KEY)


if __name__ == "__main__":
    main()