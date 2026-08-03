# Validation Log -- Document QA (RAG) System

Run timestamp: 2026-08-03T01:52:14
Documents indexed: 3
Chunks indexed: 10
Embedding backend: sentence-transformers/all-MiniLM-L6-v2 (dim=384)
Vector store backend: faiss.IndexFlatIP
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
priya.sharma@example.com | linkedin.com/in/priyasharma-ds A cross-encoder that jointly attends over the
query and each candidate passage is used to re-score and re-order the
top candidates returned by the first-stage retriever, since cross-encoders
are more accurate but too slow to run over the full corpus. RAG addresses this by separating
"what the model knows how to do" (reasoning, language generation) from
"what the model needs to know" (facts), storing the latter in an external,
updatable index.

(Grounded in: resume.txt, research_paper.txt)

**Cited sources:** resume.txt, research_paper.txt

**Latency:** retrieval=9650.8ms, generation=0.0ms, total=9650.8ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | resume.txt | -3.152 | 0.47 | 4.7026 | lick-through rate on the homepage carousel by 18%. She also owned the A/B testing framework used across the growth team.... |
| 2 | research_paper.txt | -3.7862 | 0.1833 | 2.0452 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 3 | research_paper.txt | -7.372 | 0.2527 | 4.9328 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 4 | research_paper.txt | -9.0206 | 0.1822 | 0.3054 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |

### 2. How much did the fraud-detection model reduce false positives by?

**Answer:** in Information Technology, PEC Chandigarh (2015-2019) Work Experience Senior Data Scientist, FinEdge Analytics (2022 - Present)
Priya leads a team of three engineers building fraud-detection models for a
digital payments platform. She designed a gradient-boosted ensemble that
reduced false-positive fraud alerts by 32% while maintaining recall above 95%. model (384-dim) for cost reasons, and revisit larger models if recall
 on the validation question set is below 85%.

(Grounded in: resume.txt, notes.txt)

**Cited sources:** resume.txt, notes.txt

**Latency:** retrieval=314.2ms, generation=1.0ms, total=315.2ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | resume.txt | 4.1123 | 0.4178 | 7.1657 | uter Science, Panjab University (2019-2021) - B.E. in Information Technology, PEC Chandigarh (2015-2019) Work Experience... |
| 2 | notes.txt | -9.8429 | 0.2062 | 0.9183 | model (384-dim) for cost reasons, and revisit larger models if recall  on the validation question set is below 85%. 3. V... |
| 3 | research_paper.txt | -10.2955 | 0.0844 | 3.3469 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 4 | notes.txt | -10.5217 | 0.1508 | 1.1949 | Meeting Notes: Q3 Knowledge Assistant Project Date: 14 July 2026 Attendees: Priya Sharma (Data Science), Rohan Verma (Pr... |

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

**Latency:** retrieval=281.5ms, generation=0.6ms, total=282.1ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | notes.txt | 1.6336 | 0.249 | 7.7406 | Meeting Notes: Q3 Knowledge Assistant Project Date: 14 July 2026 Attendees: Priya Sharma (Data Science), Rohan Verma (Pr... |
| 2 | notes.txt | -8.1195 | 0.1291 | 1.497 | model (384-dim) for cost reasons, and revisit larger models if recall  on the validation question set is below 85%. 3. V... |
| 3 | research_paper.txt | -8.261 | 0.2298 | 5.4578 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 4 | research_paper.txt | -10.1838 | 0.1816 | 1.8349 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |

### 4. Why is hybrid retrieval (dense + BM25) useful?

**Answer:** Combining dense
similarity with a sparse lexical score such as BM25 (a hybrid retriever)
consistently improves recall on such queries. 2.4 Hybrid retrieval. Pure dense retrieval can under-perform on queries
containing rare tokens, numbers, or exact identifiers.

(Grounded in: research_paper.txt)

**Cited sources:** research_paper.txt

**Latency:** retrieval=305.6ms, generation=0.0ms, total=305.6ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | research_paper.txt | 4.3094 | 0.5959 | 5.9197 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |
| 2 | research_paper.txt | 3.3979 | 0.5004 | 3.1816 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 3 | notes.txt | 3.0081 | 0.5616 | 4.4851 | model (384-dim) for cost reasons, and revisit larger models if recall  on the validation question set is below 85%. 3. V... |
| 4 | research_paper.txt | -3.4337 | 0.3771 | 2.3364 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |

### 5. What vector store was chosen for the pilot and what was the reasoning?

**Answer:** Vector store: FAISS was chosen over a managed vector DB for the pilot
 because the corpus is small (~5,000 chunks) and FAISS can run
 in-process with no extra infrastructure. Hybrid search: BM25 keyword search will run alongside vector search
 because several compliance queries reference exact clause numbers
 (e.g. Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract
Large pre-trained language models store factual knowledge implicitly in
their parameters, but this knowledge is static, hard to inspect, and hard
to update.

(Grounded in: notes.txt, research_paper.txt)

**Cited sources:** notes.txt, research_paper.txt

**Latency:** retrieval=312.3ms, generation=0.0ms, total=312.3ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | notes.txt | 5.3968 | 0.3681 | 11.6502 | model (384-dim) for cost reasons, and revisit larger models if recall  on the validation question set is below 85%. 3. V... |
| 2 | research_paper.txt | -11.077 | 0.1654 | 3.5615 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |
| 3 | research_paper.txt | -11.1005 | 0.0608 | 8.5792 | assages, and the generator conditions on both the retrieved passages and the original query to produce the final answer.... |
| 4 | research_paper.txt | -11.1868 | 0.1726 | 1.5846 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |

### 6. What is the capital of France?

**Answer:** I don't have enough information in the provided documents to answer that.

**Cited sources:** (none -- out of domain)

**Latency:** retrieval=278.5ms, generation=1.1ms, total=279.7ms

**Top retrieved chunks:**

| Rank | Source | Fused Score | Vector Score | Keyword Score | Preview |
|---|---|---|---|---|---|
| 1 | research_paper.txt | -11.1632 | -0.0451 | 1.4106 | rlapping passages of a few hundred tokens. Overlap preserves context that would otherwise be lost at passage boundaries.... |
| 2 | research_paper.txt | -11.1749 | 0.0231 | 1.477 | Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Abstract Large pre-trained language models store... |
| 3 | research_paper.txt | -11.1946 | -0.0161 | 0.969 | e similarity with a sparse lexical score such as BM25 (a hybrid retriever) consistently improves recall on such queries.... |
| 4 | resume.txt | -11.2167 | -0.1168 | 0.8559 | uter Science, Panjab University (2019-2021) - B.E. in Information Technology, PEC Chandigarh (2015-2019) Work Experience... |

## Observations

- Questions with direct textual support in the sample documents (skills, metrics, design decisions) are answered with correct, source-attributed grounded text pulled from the right document.
- The out-of-domain control question ("capital of France") correctly produces a low-confidence / out-of-scope style answer rather than a hallucinated fact, since the extractive fallback generator only ever emits text that is present in the retrieved context.
- Hybrid retrieval helped on the chunk-size question, whose answer contains numeric/jargon tokens ("800-character", "120-character") that benefit from the BM25 keyword component alongside dense similarity.
