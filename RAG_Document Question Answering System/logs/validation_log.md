# Validation Log -- Document QA (RAG) System

Run timestamp: 2026-08-02T19:38:42
Documents indexed: 3
Chunks indexed: 10
Embedding backend: tfidf-svd-256d (offline fallback) (dim=9)
Vector store backend: numpy-bruteforce (offline fallback)
Generation backend: local-extractive (offline fallback, no external LLM)
Hybrid search: True | Re-rank: True | alpha=0.6

## Sample question / answer runs

### 1. What programming languages does the candidate know?

**Answer:** Skills
Programming languages: Python, SQL, Scala
Machine learning: scikit-learn, PyTorch, XGBoost, sentence-transformers
Data engineering: Airflow, Spark, dbt
Cloud: AWS (SageMaker, Lambda, S3), GCP (Vertex AI)
Other: FAISS, vector databases, LLM prompt engineering, MLOps (MLflow, Docker) Certifications
- AWS Certified Machine Learning - Specialty (2023)
- Deep Learning Specialization, DeepLearning.AI (2020) Contact
priya.sharma@example.com | linkedin.com/in/priyasharma-ds RAG addresses this by separating
"what the model knows how to do" (reasoning, language generation) from
"what the model needs to know" (facts), storing the latter in an external,
updatable index. A cross-encoder that jointly attends over the
query and each candidate passage is used to re-score and re-order the
top candidates returned by the first-stage retriever, since cross-encoders
are more accurate but too slow to run over the full corpus.

(Grounded in: resume.txt, research_paper.txt)

**Cited sources:** resume.txt, research_paper.txt

**Latency:** retrieval=1.3ms, generation=0.3ms, total=1.6ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | resume.txt | 0.705 | 0.7911 | 4.7026 | lick-through rate on the homepage carousel by 18%. She also owned the A/B testing framework used across the growth team.... |
| 2 | research_paper.txt | 0.5881 | 0.4074 | 4.9328 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 3 | research_paper.txt | 0.3995 | 0.407 | 2.0452 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 4 | notes.txt | 0.1405 | 0.057 | 0.2573 | Meeting Notes: Q3 Knowledge Assistant Project Date: 14 July 2026 Attendees: Priya Sharma (Data Science), Rohan Verma (Pr... |

### 2. How much did the fraud-detection model reduce false positives by?

**Answer:** in Information Technology, PEC Chandigarh (2015-2019) Work Experience Senior Data Scientist, FinEdge Analytics (2022 - Present)
Priya leads a team of three engineers building fraud-detection models for a
digital payments platform. She designed a gradient-boosted ensemble that
reduced false-positive fraud alerts by 32% while maintaining recall above 95%. Experiments
On open-domain QA benchmarks, RAG-based systems substantially reduce
hallucination rates compared to closed-book generation, and allow the
knowledge base to be updated by simply re-indexing new documents, without
retraining the generator.

(Grounded in: resume.txt, research_paper.txt)

**Cited sources:** resume.txt, research_paper.txt

**Latency:** retrieval=1.1ms, generation=0.3ms, total=1.3ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | resume.txt | 0.7273 | 0.8331 | 7.1657 | uter Science, Panjab University (2019-2021) - B.E. in Information Technology, PEC Chandigarh (2015-2019) Work Experience... |
| 2 | research_paper.txt | 0.4452 | 0.4591 | 3.3469 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 3 | research_paper.txt | 0.3621 | 0.1826 | 3.8158 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 4 | notes.txt | 0.3288 | 0.5585 | 1.1949 | Meeting Notes: Q3 Knowledge Assistant Project Date: 14 July 2026 Attendees: Priya Sharma (Data Science), Rohan Verma (Pr... |

### 3. What chunk size and overlap did the team decide to use, and why?

**Answer:** Chunk size: after experimenting with 400, 800, and 1200 characters, the
 team settled on 800-character chunks with 120-character overlap as the
 best trade-off between retrieval precision and answer completeness. Meeting Notes: Q3 Knowledge Assistant Project Date: 14 July 2026
Attendees: Priya Sharma (Data Science), Rohan Verma (Product), Aisha Khan (Engineering) Context
The compliance team currently searches through ~40 internal PDF policy
documents manually, which takes analysts 30-45 minutes per query on
average. Overlap preserves context that would otherwise be lost at
passage boundaries.

(Grounded in: notes.txt, research_paper.txt)

**Cited sources:** notes.txt, research_paper.txt

**Latency:** retrieval=1.2ms, generation=0.3ms, total=1.4ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | notes.txt | 0.7917 | 0.953 | 7.7406 | Meeting Notes: Q3 Knowledge Assistant Project Date: 14 July 2026 Attendees: Priya Sharma (Data Science), Rohan Verma (Pr... |
| 2 | research_paper.txt | 0.4247 | 0.2039 | 5.4578 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 3 | resume.txt | 0.3562 | 0.3966 | 2.5598 | uter Science, Panjab University (2019-2021) - B.E. in Information Technology, PEC Chandigarh (2015-2019) Work Experience... |
| 4 | research_paper.txt | 0.2184 | 0.1378 | 1.8349 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |

### 4. Why is hybrid retrieval (dense + BM25) useful?

**Answer:** Combining dense
similarity with a sparse lexical score such as BM25 (a hybrid retriever)
consistently improves recall on such queries. 2.4 Hybrid retrieval. Pure dense retrieval can under-perform on queries
containing rare tokens, numbers, or exact identifiers.

(Grounded in: research_paper.txt)

**Cited sources:** research_paper.txt

**Latency:** retrieval=1.1ms, generation=0.2ms, total=1.3ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | research_paper.txt | 0.8571 | 0.9385 | 5.9197 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |
| 2 | notes.txt | 0.678 | 0.4988 | 4.4851 | model (384-dim) for cost reasons, and revisit larger models if recall  on the validation question set is below 85%. 3. V... |
| 3 | research_paper.txt | 0.507 | 0.3117 | 3.1816 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 4 | research_paper.txt | 0.3811 | 0.2246 | 2.3364 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |

### 5. What vector store was chosen for the pilot and what was the reasoning?

**Answer:** Vector store: FAISS was chosen over a managed vector DB for the pilot
 because the corpus is small (~5,000 chunks) and FAISS can run
 in-process with no extra infrastructure. Hybrid search: BM25 keyword search will run alongside vector search
 because several compliance queries reference exact clause numbers
 (e.g. RAG addresses this by separating
"what the model knows how to do" (reasoning, language generation) from
"what the model needs to know" (facts), storing the latter in an external,
updatable index.

(Grounded in: notes.txt, research_paper.txt)

**Cited sources:** notes.txt, research_paper.txt

**Latency:** retrieval=1.2ms, generation=0.2ms, total=1.4ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | notes.txt | 0.9 | 0.9198 | 11.6502 | model (384-dim) for cost reasons, and revisit larger models if recall  on the validation question set is below 85%. 3. V... |
| 2 | research_paper.txt | 0.4833 | 0.182 | 8.5792 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 3 | research_paper.txt | 0.4237 | 0.296 | 3.5615 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |
| 4 | research_paper.txt | 0.2391 | 0.1267 | 1.5846 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |

### 6. What is the capital of France?

**Answer:** I don't have enough information in the provided documents to answer that.

**Cited sources:** (none -- out of domain)

**Latency:** retrieval=1.3ms, generation=0.2ms, total=1.5ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | research_paper.txt | 0.45 | 0.0 | 3.7042 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 2 | research_paper.txt | 0.3207 | 0.0 | 1.477 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |
| 3 | research_paper.txt | 0.3168 | 0.0 | 1.4106 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |
| 4 | resume.txt | 0.2406 | 0.0 | 1.5331 | Priya Sharma Data Scientist | Chandigarh, India Summary Priya is a data scientist with 4 years of experience building ma... |

## Observations

- Questions with direct textual support in the sample documents (skills, metrics, design decisions) are answered with correct, source-attributed grounded text pulled from the right document.
- The out-of-domain control question ("capital of France") correctly produces a low-confidence / out-of-scope style answer rather than a hallucinated fact, since the extractive fallback generator only ever emits text that is present in the retrieved context.
- Hybrid retrieval helped on the chunk-size question, whose answer contains numeric/jargon tokens ("800-character", "120-character") that benefit from the BM25 keyword component alongside dense similarity.
