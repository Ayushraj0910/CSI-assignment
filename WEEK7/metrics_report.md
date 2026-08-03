# System Metrics Report -- Document QA (RAG) System

Generated: 2026-08-03T01:52:14

## 1. Chunking Profile

- Strategy: paragraph/sentence-aware recursive splitting with hard-wrap fallback
- Chunk size: 800 characters
- Chunk overlap: 120 characters
- Total chunks produced: 10 (from 3 source documents)

## 2. Embedding Configuration

- Backend: sentence-transformers/all-MiniLM-L6-v2
- Embedding dimension: 384
- Primary (production) backend: sentence-transformers `all-MiniLM-L6-v2` (384-dim)
- Offline fallback backend: TF-IDF + Truncated SVD dense projection

## 3. Vector Store

- Backend: faiss.IndexFlatIP
- Similarity metric: inner product on L2-normalized vectors (== cosine similarity)
- Production recommendation: FAISS `IndexFlatIP` (exact) or `IndexIVFFlat` (approximate, for larger corpora)

## 4. Retrieval Configuration

- top_k: 4
- Hybrid search (dense + BM25 keyword): True
- Fusion weight alpha (vector vs keyword): 0.6
- Re-ranking layer enabled: True
- Re-ranker: cross-encoder `ms-marco-MiniLM-L-6-v2` when available, else lexical-overlap heuristic blended with fused score

## 5. Generation / LLM Setup

- Backend used in this run: local-extractive (offline fallback, no external LLM)
- Production backends supported: Anthropic Claude (`claude-sonnet-4-6`) or OpenAI (`gpt-4o-mini`) via API key auto-detection
- Offline fallback: hallucination-free extractive generator (selects and stitches the most query-relevant sentences directly from retrieved context)
- Prompt template: system instruction constraining answers strictly to retrieved context, with explicit fallback phrase for unanswerable questions, and source citation requirement

## 6. Timing Breakdown (this run)

- Ingestion: 0.5 ms
- Chunking: 0.0 ms
- Embedding (full corpus): 290.3 ms
- Indexing: 204.8 ms

## 7. Experiments / Optimizations Explored

| Experiment | Setting A | Setting B | Observation |
|---|---|---|---|
| Chunk size | 400 chars | 800 chars | 800 chars retained more complete sentences per chunk and reduced fragmented context for multi-clause answers |
| Retrieval mode | Vector-only | Hybrid (vector + BM25) | Hybrid improved retrieval on queries with exact numbers/jargon (e.g. chunk-size figures, section references) |
| Re-ranking | Off | On | Re-ranking corrected ordering when two chunks had similar vector scores but different lexical relevance to the question |
