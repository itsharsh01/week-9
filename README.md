# NCERT RAG Q&A System — Week 9 Mini Project

A **Retrieval-Augmented Generation (RAG)** pipeline that answers CBSE Class 9 science questions grounded entirely in the official NCERT textbook PDF.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Source Document** | NCERT Class 9 Science Textbook (`ncert-9.pdf`) |
| **Retrieval Method** | BM25 (sparse lexical retrieval via `rank-bm25`) |
| **Chunking Strategy** | Token-bounded sliding window (T5 tokenizer, size=25, overlap=10) |
| **LLM Backend** | Gemini 1.5 Flash (via REST API) |
| **Grounding** | Answer-only-from-context prompt with low-confidence fallback |

---

## 🗂️ Project Structure

```
week-9/
├── final_solution.py        # Main RAG pipeline (all steps)
├── notebook.ipynb           # Step-by-step Jupyter walkthrough
├── requirements.txt         # Python dependencies
├── evaluation_results.md    # Per-question RAG evaluation
├── reflection.md            # Design decisions & lessons learned
├── failure_modes.md         # Known failure patterns & mitigations
├── .env.example             # Template for environment variables
└── ncert-9.pdf              # Source textbook (not committed to git)
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

### 2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Set your API key
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "your_api_key_here"

# Linux / macOS
export GEMINI_API_KEY="your_api_key_here"
```
Or copy `.env.example` to `.env` and fill in your key (if using a dotenv loader).

### 4. Run
```powershell
python final_solution.py
```

---

## 🔄 Pipeline Architecture

```
PDF
 │
 ▼
extract_text_from_pdf()      ← pdfplumber (pure Python)
 │
 ▼
build_structured_blocks()    ← boundary detection + block classification
 │   (question / example / formula / concept)
 ▼
chunk_blocks()               ← T5 tokenizer sliding window (size=25, overlap=10)
 │
 ▼
build_bm25_index()           ← BM25Okapi sparse index
 │
 ▼
retrieve(query, top_k=5)     ← ranked chunk retrieval
 │
 ▼
is_low_confidence()          ← score threshold check (< 0.5)
 │
 ▼
build_prompt()               ← grounded context prompt
 │
 ▼
generate_answer()            ← Gemini 1.5 Flash REST call
 │
 ▼
Printed Q&A output
```

---

## 📋 CBSE Questions Evaluated

1. What are the characteristics of particles of matter?
2. What is the difference between a mixture and a compound?
3. State the law of conservation of mass.
4. What is the difference between speed and velocity?
5. State Newton's second law of motion.
6. What is the universal law of gravitation?
7. What is kinetic energy? Give its formula.
8. What is the difference between loudness and intensity of sound?
9. What are the differences between infectious and non-infectious diseases?
10. What is the role of the atmosphere in climate control?

---

## ⚠️ Known Limitations

- **BM25 is lexical** — paraphrased queries may miss relevant chunks.
- **Chunk size = 25 tokens** is small; multi-sentence answers may be split.
- **pdfplumber** emits encoding warnings on some NCERT pages — these are non-fatal.
- **Low-confidence detection** uses a fixed score threshold and may misclassify.

See [`failure_modes.md`](failure_modes.md) for detailed analysis.

---

## 📄 License

For educational use only. NCERT content is © NCERT, Government of India.
