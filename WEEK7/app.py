"""
app.py
======
Interactive Document Question Answering (RAG) CLI.

Point this at your own documents and ask your own questions, live, from
the terminal. This is the main entrypoint for actually *using* the system
(as opposed to `validate.py`, which runs a fixed batch of test questions
and writes the validation/metrics reports for evaluation purposes).

Usage
-----
    # Use the bundled sample_data/ folder
    python app.py

    # Point at your own folder of .txt/.md/.pdf files
    python app.py --docs /path/to/your/documents

    # Point at a single file
    python app.py --docs /path/to/resume.pdf

    # Tune retrieval behavior
    python app.py --docs ./my_notes --chunk-size 600 --top-k 5 --no-rerank

Once the index is built, you'll get an interactive prompt:

    Ask a question (or 'quit' to exit):
    > What is the candidate's most recent job?

    Answer: ...
    Sources: [...]

Type 'quit', 'exit', or press Ctrl+C to stop.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import RAGPipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive RAG Document Question Answering system."
    )
    parser.add_argument(
        "--docs",
        default=os.path.join(os.path.dirname(__file__), "sample_data"),
        help="Path to a directory of .txt/.md/.pdf files, or a single file. "
             "Defaults to the bundled sample_data/ folder.",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters (default: 120)")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve per question (default: 4)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Vector vs keyword fusion weight, 0-1 (default: 0.6)")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid (vector + BM25) retrieval")
    parser.add_argument("--no-rerank", action="store_true", help="Disable the re-ranking layer")
    parser.add_argument("--question", "-q", default=None, help="Ask a single question and exit (non-interactive mode)")
    return parser.parse_args()


def build_pipeline(args) -> RAGPipeline:
    rag = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        hybrid=not args.no_hybrid,
        rerank=not args.no_rerank,
        alpha=args.alpha,
    )

    docs_path = args.docs
    if not os.path.exists(docs_path):
        print(f"Error: path not found: {docs_path}")
        sys.exit(1)

    print(f"Ingesting documents from: {docs_path}")
    if os.path.isdir(docs_path):
        rag.ingest_directory(docs_path)
    else:
        rag.ingest_file(docs_path)

    if not rag.documents:
        print("No supported documents found (.txt, .md, .pdf). Nothing to index.")
        sys.exit(1)

    print(f"  -> Loaded {len(rag.documents)} document(s): "
          + ", ".join(d.source for d in rag.documents))

    print("Chunking + embedding + indexing ...")
    rag.build_index()
    report = rag.system_report()
    print(f"  -> {report['num_chunks']} chunks | "
          f"embeddings: {report['embedding_backend']} (dim={report['embedding_dim']}) | "
          f"vector store: {report['vector_store_backend']} | "
          f"generation: {report['generation_backend']}")
    print()
    return rag


def print_answer(rag: RAGPipeline, question: str):
    result = rag.ask(question)
    print()
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"(retrieval={result['retrieval_sec']*1000:.1f}ms, generation={result['generation_sec']*1000:.1f}ms)")
    print()


def run():
    args = parse_args()
    rag = build_pipeline(args)

    if args.question:
        # Non-interactive single-shot mode
        print_answer(rag, args.question)
        return

    print("=" * 70)
    print("Ready. Ask a question about your documents (or type 'quit' to exit).")
    print("=" * 70)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        print_answer(rag, question)


if __name__ == "__main__":
    run()
