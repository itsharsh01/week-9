# Failure Modes — NCERT RAG Q&A System (Advanced)

A structured post-mortem of every failure class observed during development and evaluation of the NCERT RAG pipeline, with root cause analysis and proposed mitigations.

---

## Failure Taxonomy

```
RAG Failures
├── Retrieval Failures
│   ├── FM-1: Lexical Mismatch (paraphrase gap)
│   ├── FM-2: Chunk Boundary Fragmentation
│   └── FM-3: Low-Density Chapter Sparsity
├── Chunking Failures
│   ├── FM-4: Over-Splitting (chunk too small)
│   └── FM-5: Block Misclassification
├── Generation Failures
│   ├── FM-6: Confident Wrong Answer (hallucination with high BM25)
│   └── FM-7: Over-Hedging (false low-confidence)
└── Infrastructure Failures
    ├── FM-8: Native DLL AppControl Block
    └── FM-9: PDF Encoding Warnings
```

---

## FM-1: Lexical Mismatch (Paraphrase Gap)

**Observed in:** Q9 (infectious diseases), Q10 (atmosphere/climate)

**Symptom:** BM25 returns low-scored chunks even though the textbook clearly covers the topic.

**Root Cause:**  
BM25 is a bag-of-words model. It requires token overlap between the query and the document. When the user asks "climate control" but the textbook says "regulation of temperature" or "greenhouse effect", BM25 scores collapse.

**Evidence:**
```
Query:     "role of atmosphere in climate control"
Top chunk score: 0.31  (below 0.5 threshold → low-confidence triggered)
Textbook terms: "greenhouse gases", "heat retention", "temperature regulation"
```

**Mitigations:**
1. **Query expansion** — Before BM25 lookup, expand the query with domain synonyms:
   ```python
   synonyms = {
       "climate control": ["greenhouse effect", "temperature regulation", "heat retention"],
       "infectious":      ["communicable", "pathogen", "contagious"],
   }
   ```
2. **Dense retrieval** — Use `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic similarity scoring, bypassing the lexical gap entirely.
3. **Hybrid BM25 + dense** — Run both, then merge scores with a linear combination: `score = α * bm25 + (1-α) * dense`.

---

## FM-2: Chunk Boundary Fragmentation

**Observed in:** Q8 (loudness vs. intensity of sound)

**Symptom:** Answer is partial — model gets one side of the comparison but not both.

**Root Cause:**  
With `chunk_size=25` tokens and `overlap=10`, a multi-sentence explanation is split across 3–4 chunks. Only 1–2 of those chunks make it into the top-5 retrieved set, so the LLM sees an incomplete picture.

**Evidence:**
```
Block (original):
  "Loudness depends on the amplitude of vibration of the sound wave...
   Intensity is the amount of energy passing per unit area per second..."

After chunking (size=25):
  Chunk A: "Loudness depends on the amplitude of vibration of the"
  Chunk B: "sound wave... Intensity is the amount of energy"
  Chunk C: "passing per unit area per second..."

Retrieved: Chunk A (score=0.82), Chunk C (score=0.61) → gap in middle
```

**Mitigations:**
1. **Increase chunk size** — Set `chunk_size=60, overlap=20` to preserve complete sentences.
2. **Parent-child chunking** — Store small chunks (size=25) for retrieval but send the larger parent block (size=100) to the LLM.
3. **Sentence-aware splitting** — Use `nltk.sent_tokenize` to split at sentence boundaries instead of raw token count.

---

## FM-3: Low-Density Chapter Sparsity

**Observed in:** Chapters 13 & 14 (Why Do We Fall Ill, Natural Resources)

**Symptom:** BM25 scores uniformly low across all retrieved chunks regardless of query.

**Root Cause:**  
Later NCERT chapters are written in narrative/descriptive prose with fewer repeated technical keywords. BM25's IDF scores inflate for common words in these chapters, causing all queries to score similarly low.

**Evidence:**
```
Ch. 1 (Matter) avg BM25 score for keyword queries: ~1.8
Ch. 14 (Natural Resources) avg BM25 score: ~0.3
```

**Mitigations:**
1. **Per-chapter BM25 index** — Build separate BM25 indices per chapter; route the query to the most relevant chapter index first.
2. **Dense embeddings** — Prose chapters benefit more from semantic similarity than keyword matching.
3. **Metadata filtering** — Tag chunks with their chapter number; allow the user to optionally scope the search.

---

## FM-4: Over-Splitting (Chunk Too Small)

**Observed in:** All multi-part questions (Q7, Q8)

**Symptom:** Formulae and their explanations end up in separate chunks, breaking the answer.

**Root Cause:**  
`chunk_size=25 T5 tokens ≈ 15–18 English words`. A typical NCERT sentence is 20–30 words. This means almost every sentence is split.

**T5 Token Count Analysis:**
```
"Kinetic energy is the energy possessed by an object due to its motion."
 → 16 T5 tokens  → fits in one chunk ✓

"The kinetic energy of an object of mass m, moving with velocity v is given by KE = ½mv²"
 → 28 T5 tokens  → split into 2 chunks ✗
```

**Mitigation:** Increase `chunk_size` to at least 50 tokens.

---

## FM-5: Block Misclassification

**Symptom:** A "concept" block is classified as "question" because it contains the word "q" (e.g., "liquid").

**Root Cause:**  
`classify_block()` uses naive substring matching:
```python
if any(k in t for k in ["question", "questions", "q"]):
    return "question"
```
The single letter `"q"` matches inside many common words.

**Affected words:** "liquid", "unique", "equal", "acquire", "require"

**Fix:**
```python
# Replace substring check with whole-word regex
import re
if re.search(r'\bquestions?\b', t) or re.search(r'\bq\.\s*\d', t):
    return "question"
```

---

## FM-6: Confident Wrong Answer (Hallucination with High BM25)

**Symptom:** BM25 retrieves a plausible-looking chunk that is topically adjacent but not the correct answer. LLM generates a confident but wrong answer.

**Root Cause:**  
BM25 scores lexical overlap, not semantic correctness. A chunk about "velocity" may score highly for a "speed" query, but they are different concepts.

**Example scenario:**
```
Query: "Define uniform acceleration."
Retrieved: chunk about "uniform motion" (high BM25 due to "uniform")
LLM output: confuses uniform motion with uniform acceleration
```

**Mitigations:**
1. **Re-ranking** — Add a cross-encoder re-ranker (`ms-marco-MiniLM`) that scores query-chunk semantic relevance.
2. **Answer verification** — After generation, run a secondary check: "Does this answer actually address the question?" as a separate Gemini call.

---

## FM-7: Over-Hedging (False Low-Confidence)

**Symptom:** Low-confidence fallback triggers even when the correct answer is in the top-5 chunks.

**Root Cause:**  
The threshold (0.5) was set empirically on chapters with high keyword density. For prose chapters, all BM25 scores hover at 0.2–0.4 even for correct retrievals.

**Fix:**  
Use a **relative threshold** — compare the top chunk score against the mean chunk score:
```python
def is_low_confidence(retrieved_chunks, z_threshold=1.0):
    scores = [c["score"] for c in retrieved_chunks]
    if not scores:
        return True
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    # Low confidence if even the top score is within 1 std of the mean
    return (scores[0] - mean) < z_threshold * std
```

---

## FM-8: Native DLL AppControl Block

**Symptom:** `ImportError: DLL load failed while importing _extra: An Application Control policy has blocked this file.`

**Affected packages:** `PyMuPDF` (fitz), `pydantic-core`, `google-generativeai`

**Root Cause:**  
Windows WDAC policy blocks unsigned native extensions (`.pyd`, `.dll`) installed in user-writable directories (`AppData`, `.venv`).

**Resolution applied:**
| Original package | Pure-Python replacement |
|---|---|
| `PyMuPDF` (fitz) | `pdfplumber` (pdfminer.six) |
| `google-generativeai` + `pydantic-core` | Direct `requests` REST call to Gemini API |

---

## FM-9: PDF Encoding Warnings

**Symptom:**
```
Could not set character spacing because b'' is an invalid float value
Cannot render horizontal string because ... is not a valid int, float or bytes.
```

**Root Cause:**  
NCERT PDFs include some pages with non-standard Type1 font encodings. `pdfminer` (used internally by `pdfplumber`) cannot parse these operator arguments.

**Impact:** Non-fatal. Affected pages return partial or empty text strings; the pipeline skips them gracefully via the `if extracted:` guard.

**Mitigation:** Suppress the warnings at runtime:
```python
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
```

---

## Summary Table

| ID | Category | Severity | Status |
|---|---|---|---|
| FM-1 | Retrieval — Paraphrase Gap | High | ⚠️ Partial — low-conf fallback mitigates |
| FM-2 | Retrieval — Fragmentation | Medium | ⚠️ Partial — chunk size increase recommended |
| FM-3 | Retrieval — Chapter Sparsity | Medium | ⚠️ Partial — low-conf fallback helps |
| FM-4 | Chunking — Over-Splitting | Medium | 🔧 Fix: increase chunk_size to 50 |
| FM-5 | Chunking — Misclassification | Low | 🔧 Fix: regex whole-word match |
| FM-6 | Generation — Hallucination | High | 🔧 Fix: re-ranker + verification |
| FM-7 | Generation — Over-Hedging | Low | 🔧 Fix: relative confidence threshold |
| FM-8 | Infrastructure — DLL Block | Critical | ✅ Resolved — pure-Python replacements |
| FM-9 | Infrastructure — PDF Encoding | Low | ✅ Mitigated — non-fatal, guarded skip |
