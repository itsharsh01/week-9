# Evaluation Results — NCERT RAG Q&A System

**Model:** Gemini 1.5 Flash  
**Retrieval:** BM25 (rank-bm25), top_k = 5  
**Chunking:** T5 tokenizer, chunk_size = 25, overlap = 10  
**Source:** NCERT Class 9 Science Textbook (`ncert-9.pdf`)  
**Grounding constraint:** Answer ONLY from retrieved context

---

## Scoring Rubric

| Score | Meaning |
|---|---|
| ✅ Correct | Answer is factually accurate and grounded in the textbook |
| ⚠️ Partial | Answer captures some correct points but is incomplete or vague |
| ❌ Incorrect | Answer is wrong, hallucinated, or irrelevant |
| 🔍 Low Conf | BM25 retrieval scored < 0.5 on top-3 chunks |

---

## Per-Question Results

### Q1 — What are the characteristics of particles of matter?

| Field | Value |
|---|---|
| **Chapter** | 1 — Matter in Our Surroundings |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | BM25 correctly retrieved the "particles of matter" section. Answer covered intermolecular spaces, kinetic energy, and inter-particle forces. |

---

### Q2 — What is the difference between a mixture and a compound?

| Field | Value |
|---|---|
| **Chapter** | 2 — Is Matter Around Us Pure |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | Retrieved chunks included definition blocks for mixture and compound. Answer correctly described fixed vs variable composition and physical vs chemical combination. |

---

### Q3 — State the law of conservation of mass.

| Field | Value |
|---|---|
| **Chapter** | 3 — Atoms and Molecules |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | Exact law statement retrieved from the "formula" classified chunk. Answer matched the textbook definition verbatim. |

---

### Q4 — What is the difference between speed and velocity?

| Field | Value |
|---|---|
| **Chapter** | 8 — Motion |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | Answer correctly identified speed as scalar and velocity as vector. Formula chunks contributed the definitions. |

---

### Q5 — State Newton's second law of motion.

| Field | Value |
|---|---|
| **Chapter** | 9 — Force and Laws of Motion |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | F = ma relationship stated correctly. Retrieved chunks were classified as "formula" type and contained the relevant equation block. |

---

### Q6 — What is the universal law of gravitation?

| Field | Value |
|---|---|
| **Chapter** | 10 — Gravitation |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | The gravitational formula F = Gm₁m₂/r² was present in retrieved chunks. Answer explained proportionality correctly. |

---

### Q7 — What is kinetic energy? Give its formula.

| Field | Value |
|---|---|
| **Chapter** | 11 — Work and Energy |
| **Low Confidence** | No |
| **Result** | ✅ Correct |
| **Notes** | KE = ½mv² retrieved correctly from "formula" chunks. Definition and formula both included in answer. |

---

### Q8 — What is the difference between loudness and intensity of sound?

| Field | Value |
|---|---|
| **Chapter** | 12 — Sound |
| **Low Confidence** | ⚠️ Possible |
| **Result** | ⚠️ Partial |
| **Notes** | BM25 retrieved "sound" chunks but the specific loudness vs. intensity distinction was spread across multiple small chunks. Answer was somewhat vague on the physiological vs. physical distinction. |

---

### Q9 — What are the differences between infectious and non-infectious diseases?

| Field | Value |
|---|---|
| **Chapter** | 13 — Why Do We Fall Ill |
| **Low Confidence** | 🔍 Yes |
| **Result** | ⚠️ Partial |
| **Notes** | Low confidence triggered the fallback prompt. The chapter content was present in the PDF but relevant chunks scored poorly under BM25 due to sparse keyword overlap with the query. Answer was generic. |

---

### Q10 — What is the role of the atmosphere in climate control?

| Field | Value |
|---|---|
| **Chapter** | 14 — Natural Resources |
| **Low Confidence** | 🔍 Yes |
| **Result** | ⚠️ Partial |
| **Notes** | Similar to Q9 — "climate control" is a paraphrase not directly present in the text. BM25 failed to retrieve highly relevant chunks. Answer relied on low-confidence fallback. |

---

## Aggregate Summary

| Metric | Value |
|---|---|
| Total questions | 10 |
| ✅ Correct | 7 (70%) |
| ⚠️ Partial | 3 (30%) |
| ❌ Incorrect | 0 (0%) |
| 🔍 Low-confidence triggered | 2 (20%) |

---

## Key Observations

1. **BM25 excels at direct keyword matches** — Questions using exact NCERT terminology (e.g., "Newton's second law", "law of conservation of mass") retrieve well.
2. **Paraphrased queries underperform** — "climate control" vs. "greenhouse effect / heat retention" causes retrieval miss.
3. **Small chunk size hurts multi-part answers** — Loudness vs. intensity requires stitching across several adjacent chunks.
4. **Low-confidence fallback works correctly** — The model did not hallucinate when confidence was low; it appropriately hedged its answer.
5. **Chapter 13–14 content is sparse in BM25** — These chapters contain longer prose with fewer distinct keywords, lowering BM25 scores.

---

## Suggested Improvements

| Issue | Proposed Fix |
|---|---|
| Paraphrase mismatch | Add semantic (dense) retrieval with `sentence-transformers` |
| Small chunks losing context | Increase `chunk_size` to 50–75 tokens |
| Low recall on prose chapters | Use hybrid BM25 + embedding re-ranking |
| Fixed threshold confidence | Use per-query adaptive confidence threshold |
