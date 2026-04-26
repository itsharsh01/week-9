# Reflection — NCERT RAG Q&A System

## What I Built

A fully local, grounded Retrieval-Augmented Generation (RAG) system that answers CBSE Class 9 science questions using **only** the official NCERT textbook as its knowledge source. The system was intentionally built without vector databases, neural embeddings, or heavy ML frameworks — to understand the fundamentals of RAG before layering on abstractions.

---

## Design Decisions & Why

### 1. BM25 over Dense Embeddings

**Decision:** Use `rank-bm25` (sparse TF-IDF-style retrieval) rather than `sentence-transformers` or a vector DB.

**Rationale:**
- BM25 is fully interpretable — I can see exactly why a chunk was retrieved.
- No GPU, no large model download, no AppControl DLL issues.
- For textbook Q&A with direct keyword matches, BM25 performs surprisingly well.

**Trade-off:** Paraphrased or conceptual queries (e.g., "climate control" instead of "greenhouse effect") miss relevant chunks entirely. This was observed in Q9 and Q10.

---

### 2. T5 Tokenizer for Chunking

**Decision:** Use the T5-small tokenizer to split blocks into fixed-size token windows (size=25, overlap=10).

**Rationale:**
- Token-level chunking is more principled than character or word splitting.
- T5's SentencePiece tokenizer handles scientific notation and formulae gracefully.
- The tokenizer itself (not the model) is downloaded — no PyTorch needed.

**Trade-off:** Chunk size of 25 tokens is very small. Multi-sentence explanations get fragmented, harming answers that need broader context (e.g., the loudness vs. intensity question).

---

### 3. Boundary-Aware Block Splitting

**Decision:** Before chunking, split the raw PDF text into semantic blocks using a rule-based boundary detector (`is_boundary()`).

**Rationale:**
- NCERT PDFs follow a predictable structure: headings, exercises, examples, and concept paragraphs.
- Respecting these boundaries means chunks don't straddle unrelated sections.
- Classifying blocks as `question / example / formula / concept` allows the prompt to signal the type of context.

**Trade-off:** The rules are hand-crafted and brittle. A heading that doesn't match the regex patterns will be merged into the wrong block.

---

### 4. REST API over SDK for Gemini

**Decision:** Call the Gemini REST API directly via `requests` instead of using `google-generativeai`.

**Rationale:**
- The `google-generativeai` SDK pulls in `pydantic-core`, which is a native C extension DLL.
- On this machine, Windows Application Control (WDAC) blocked all native DLLs installed in user-writable directories.
- `requests` is pure Python and installed in the system's trusted site-packages.

**Lesson:** Always check for native DLL dependencies when working in managed/enterprise environments. Pure-Python alternatives often exist.

---

### 5. Confidence Gating

**Decision:** Check if the top-3 retrieved chunks all score below 0.5; if so, use a more cautious prompt.

**Rationale:**
- Without confidence gating, the LLM would confidently answer from irrelevant chunks.
- The low-confidence prompt instructs the model to hedge or say "I don't know."
- This prevents hallucination at the cost of occasionally under-answering.

**Trade-off:** A single fixed threshold (0.5) is too blunt. Q8 (loudness vs. intensity) had borderline scores that were not caught.

---

## What Worked Well

- The pipeline ran **end-to-end** with zero GPU requirements.
- BM25 correctly retrieved relevant chunks for **7/10 questions** on the first attempt.
- The grounding constraint ("answer ONLY from context") effectively prevented hallucination.
- Block classification gave the prompt useful metadata about chunk type.
- The low-confidence fallback correctly hedged on Q9 and Q10.

---

## What Didn't Work (and Why)

| Problem | Root Cause | Impact |
|---|---|---|
| Q8 partial answer | Chunk size too small; loudness/intensity explanation split across chunks | ⚠️ Partial |
| Q9 low recall | "infectious vs. non-infectious" poorly matched by BM25 | 🔍 Low confidence |
| Q10 low recall | "climate control" is not the NCERT keyword; "atmosphere" chunks too broad | 🔍 Low confidence |
| PDF encoding warnings | NCERT PDF uses non-standard font encoding in some pages | Non-fatal, noise |
| DLL blocks | AppControl policy blocked PyMuPDF and pydantic_core | Forced library switches |

---

## Lessons Learned

1. **Start simple.** BM25 + a grounded LLM prompt gets you further than expected for structured-document Q&A.
2. **Environment constraints drive architecture.** WDAC policies forced better choices (pure Python, REST API).
3. **Chunk size matters more than retrieval algorithm** for questions requiring multi-sentence context.
4. **Confidence scores need calibration.** A single threshold doesn't generalize across chapters with different vocabulary density.
5. **Block classification adds real value** — knowing a chunk is a "formula" vs. "concept" changes how the LLM weighs it.

---

## If I Were to Rebuild This

1. **Hybrid retrieval** — BM25 for recall + `sentence-transformers` re-ranker for precision.
2. **Larger chunks with parent-child linking** — small chunks for retrieval, large parent chunk sent to LLM.
3. **Query expansion** — expand the user query with NCERT-specific synonyms before BM25 lookup.
4. **Evaluation harness** — automated ROUGE/BERTScore comparison against textbook answer keys.
5. **Streaming output** — stream the Gemini response token-by-token for better UX.
