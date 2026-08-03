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

## 2. How to run it

There are two entrypoints:

- **`app.py`** — the interactive system. Point it at your own documents and
  ask your own questions, live, from the terminal. **Use this one.**
- **`validate.py`** — runs a fixed batch of test questions against the
  bundled `sample_data/` and regenerates `logs/validation_log.md` and
  `metrics_report.md` (the evaluation artifacts).

Neither is a program you double-click — both need to be run from a
terminal with Python. Here's the full path from zero:

### Step 1: Open a terminal inside the `rag_qa_system` folder

**Windows:**
1. Open the `rag_qa_system` folder in File Explorer.
2. Click the address bar at the top, type `cmd`, and press Enter. This
   opens a terminal already pointed at that folder.

**Mac:**
1. Right-click the `rag_qa_system` folder → **New Terminal at Folder**
   (or open the Terminal app and run `cd path/to/rag_qa_system`).

**Using VS Code instead (works on any OS):**
1. Install [VS Code](https://code.visualstudio.com/) if you don't have it.
2. File → Open Folder → select `rag_qa_system`.
3. Open the built-in terminal: **Terminal → New Terminal**.

### Step 2: Check Python is installed

```bash
python3 --version
```

If that fails, try `python --version` instead — some systems alias it
differently. You need Python 3.9+.

### Step 3: (Recommended) create a virtual environment

Keeps this project's dependencies separate from everything else on your
machine:

```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

Note: `sentence-transformers`, `faiss-cpu`, `anthropic`, `openai`, etc. are
optional production upgrades — the app still runs even if some of these
fail to install, since every stage automatically falls back to a fully
offline implementation (see section 3 below).

### Step 5: Run the interactive system

```bash
python app.py
```
(use `python3 app.py` if `python` isn't recognized)

This ingests the bundled `sample_data/` folder by default, builds the
index, then drops you into an interactive prompt:

```
Ready. Ask a question about your documents (or type 'quit' to exit).

> What programming languages does the candidate know?

Answer: ...
Sources: ['resume.txt']

> quit
Exiting.
```

Type `quit`, `exit`, or press Ctrl+C to stop.

### Using your own documents

```bash
python app.py --docs /path/to/your/documents      # a folder of .txt/.md/.pdf
python app.py --docs /path/to/resume.pdf           # or a single file
```

### Other options

```bash
python app.py --docs ./my_notes --chunk-size 600 --top-k 5 --no-rerank
python app.py -q "What is this document about?"    # ask one question and exit, no interactive loop
```

| Flag | Meaning | Default |
|---|---|---|
| `--docs` | Folder or file to ingest | `sample_data/` |
| `--chunk-size` | Chunk size in characters | 800 |
| `--chunk-overlap` | Overlap between chunks | 120 |
| `--top-k` | Chunks retrieved per question | 4 |
| `--alpha` | Vector vs keyword fusion weight (0-1) | 0.6 |
| `--no-hybrid` | Disable BM25 hybrid search | hybrid on |
| `--no-rerank` | Disable the re-ranking layer | rerank on |
| `-q` / `--question` | Ask a single question and exit (no loop) | interactive mode |

### Step 6 (optional): use a real LLM instead of the offline fallback

If you have an Anthropic or OpenAI API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows cmd
```

Then run `python app.py` again — no code changes needed, it's
auto-detected.

### Regenerating the validation log / metrics report

```bash
python validate.py
```

This runs the fixed 6-question test set against `sample_data/` and
(re)writes `logs/validation_log.md` and `metrics_report.md`.

### Using the pipeline directly from code

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
python app.py          # or: python validate.py
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

Run `python validate.py` to regenerate `logs/validation_log.md`, which contains,
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
├── app.py                    # interactive CLI -- ask your own questions on your own docs
├── validate.py                # fixed-batch evaluation runner (regenerates the reports below)
├── requirements.txt
├── README.md
├── metrics_report.md         # generated by validate.py
├── logs/
│   └── validation_log.md     # generated by validate.py
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
