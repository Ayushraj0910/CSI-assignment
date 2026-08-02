# Document Question Answering System (RAG)

A modular, end-to-end Retrieval-Augmented Generation pipeline for answering
questions over custom documents (notes, resumes, research papers, books, or
any Hugging Face text dataset such as
[`vectara/open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench)).

Instead of relying on a language model's frozen internal knowledge, the
system retrieves relevant passages from *your* documents and generates an
answer grounded in that retrieved text.

## 1. Architecture

```
   ┌─────────────┐   ┌───────────┐   ┌────────────┐   ┌───────────────┐
   │  Ingestion  │──▶│ Chunking  │──▶│ Embedding  │──▶│ Vector Store  │
   │ (txt/pdf/HF)│   │(recursive)│   │(bi-encoder)│   │ (FAISS/numpy) │
   └─────────────┘   └───────────┘   └────────────┘   └───────┬───────┘
                                                               │
   ┌─────────────┐   ┌───────────────┐   ┌─────────────────┐  │
   │   Answer    │◀──│  Generation   │◀──│    Retrieval    │◀─┘
   │ (grounded,  │   │ (LLM prompt / │   │ (vector + BM25  │
   │  cited)     │   │  extractive)  │   │  + re-ranking)  │◀── user query
   └─────────────┘   └───────────────┘   └─────────────────┘
```

Each stage lives in its own module under `src/` and shares a plain-Python
interface, so any single stage (embedding model, vector store, LLM) can be
swapped without touching the others.

| Stage | Module | What it does |
|---|---|---|
| 1. Document ingestion | `src/ingestion.py` | Loads `.txt`/`.md`, `.pdf` (via `pypdf`), or Hugging Face dataset rows into a common `Document` object |
| 2. Text chunking | `src/chunking.py` | Cleans text and splits it into overlapping, paragraph/sentence-aware chunks |
| 3. Embedding | `src/embeddings.py` | Encodes chunks into dense vectors (`sentence-transformers`, falls back to TF-IDF+SVD offline) |
| 4. Vector database | `src/vectorstore.py` | Stores vectors + metadata; cosine/inner-product similarity search (FAISS, falls back to numpy brute-force) |
| 5. Query processing | `src/retrieval.py` (`embed_query`) | Converts the user's question into the same vector space |
| 6. Retrieval | `src/retrieval.py` (`retrieve`) | Vector search + optional BM25 hybrid fusion + optional cross-encoder re-ranking |
| 7. Answer generation | `src/generation.py` | Builds a single grounded prompt from query + retrieved chunks, calls an LLM (or offline extractive fallback), returns a cited answer |
| 8. Orchestration | `src/pipeline.py` | `RAGPipeline` wires all of the above into `.ingest_*()`, `.build_index()`, `.ask()` |

## 2. Quick start

```bash
pip install -r requirements.txt   # optional extras auto-detected at runtime
python app.py
```

`app.py` ingests everything in `sample_data/` (a resume, a research paper,
and a meeting-notes file — deliberately different document *types* to show
the ingestion module handling mixed domains), builds the index, and runs a
batch of validation questions end-to-end, printing grounded answers with
source citations and timing, and writing:

- `logs/validation_log.md` — per-question answers, cited sources, and the
  top retrieved chunks with their vector/keyword/fused scores
- `metrics_report.md` — chunking profile, embedding dimensions, vector
  store choice, generation backend, and timing breakdown

### Minimal code usage

```python
import sys; sys.path.insert(0, "src")
from pipeline import RAGPipeline

rag = RAGPipeline(chunk_size=800, chunk_overlap=120, top_k=4, hybrid=True, rerank=True)
rag.ingest_directory("sample_data")     # or rag.ingest_file("my_notes.pdf")
rag.build_index()

result = rag.ask("What programming languages does the candidate know?")
print(result["answer"])
print(result["sources"])
```

### Ingesting a Hugging Face dataset

```python
from ingestion import load_hf_source
docs = load_hf_source("vectara/open_ragbench", split="train", text_field="text", max_records=200)
rag.documents.extend(docs)
rag.build_index()
```

## 3. Backend strategy: this runs anywhere, and upgrades automatically

This sandbox has no internet access, so heavyweight ML dependencies
(`sentence-transformers`, `faiss`, `transformers`, `datasets`, hosted LLM
APIs) could not be downloaded here. Rather than fake the output, every
stage that normally needs one of those dependencies ships with **a real,
fully-offline fallback implementation**, auto-selected at runtime:

| Stage | Production backend | Offline fallback (used in this demo run) |
|---|---|---|
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim) | TF-IDF + Truncated SVD dense projection |
| Vector store | FAISS `IndexFlatIP` | NumPy brute-force cosine similarity |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lexical-overlap heuristic blended with fused score |
| Answer generation | Anthropic Claude or OpenAI GPT (auto-detected via `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) | Hallucination-free extractive generator (selects and stitches the most query-relevant sentences directly from retrieved text) |

**To use the production backends**, just `pip install -r requirements.txt`
in an environment with internet access and set an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

No code changes are needed — every module tries the real backend first and
only drops to the offline version if the import/model/API key is
unavailable, so the exact same `RAGPipeline` class is what you'd run in
production.

## 4. Requirement -> implementation map

| Assignment requirement | Where it's implemented |
|---|---|
| 1. Document ingestion module (PDF / text / HF archives) | `src/ingestion.py` |
| 2. Chunking methodology | `src/chunking.py` (recursive paragraph/sentence-aware splitter + overlap) |
| 3. Chunk -> vector embeddings | `src/embeddings.py` |
| 4. Vector DB init + fast similarity search | `src/vectorstore.py` |
| 5. User query -> query vector | `src/retrieval.py::Retriever.embed_query` |
| 6. Retrieval module | `src/retrieval.py::Retriever.retrieve` |
| 7. Context + query -> LLM prompt -> grounded answer | `src/generation.py` |
| 8. Optimization experiments (chunk borders, hybrid search, re-ranking) | `src/retrieval.py` (`hybrid`, `alpha`, `rerank` parameters) + `metrics_report.md` section 7 |

## 5. Experiments performed (system optimization)

- **Chunk size sweep**: 400 / 800 / 1200 characters — 800 with 120-char
  overlap gave the best balance of retrieval precision (chunk stays on
  one topic) vs. answer completeness (full sentences/claims are not
  truncated mid-chunk). See `metrics_report.md`.
- **Hybrid retrieval (vector + BM25)**: enabled by default (`alpha=0.6`).
  Meaningfully helps queries containing exact numbers, section
  references, or jargon that a dense embedding under-weights (e.g. "800
  characters", "Section 4.2.1").
- **Re-ranking layer**: a second, independent relevance signal
  (cross-encoder in production, lexical-overlap heuristic offline) that
  re-orders the fused candidate list before it's passed to generation.

All three knobs (`chunk_size`, `hybrid`/`alpha`, `rerank`) are constructor
arguments on `RAGPipeline`, so they can be A/B compared directly.

## 6. Validation

Run `python app.py` to regenerate `logs/validation_log.md`, which contains,
for six dynamic sample questions (five in-domain, one deliberately
out-of-domain "capital of France" control question):

- the generated answer
- cited source document(s)
- retrieval/generation latency
- the ranked list of retrieved chunks with vector score, keyword score,
  and fused score

The out-of-domain control question is included specifically to validate
that the system declines to answer rather than hallucinating when the
document set genuinely doesn't contain the answer.

## 7. Project layout

```
rag_qa_system/
├── app.py                   # end-to-end demo runner
├── requirements.txt
├── README.md
├── metrics_report.md         # generated by app.py
├── logs/
│   └── validation_log.md     # generated by app.py
├── sample_data/
│   ├── resume.txt
│   ├── research_paper.txt
│   └── notes.txt
└── src/
    ├── ingestion.py
    ├── chunking.py
    ├── embeddings.py
    ├── vectorstore.py
    ├── retrieval.py
    ├── generation.py
    └── pipeline.py
```
