"""
pipeline.py
===========
End-to-end RAG orchestrator wiring together every stage:

  ingestion -> chunking -> embedding -> vector store -> retrieval -> generation

Usage
-----
    from pipeline import RAGPipeline

    rag = RAGPipeline(chunk_size=800, chunk_overlap=120, top_k=4)
    rag.ingest_directory("sample_data")
    rag.build_index()
    result = rag.ask("What programming languages does the candidate know?")
    print(result["answer"])
"""

from __future__ import annotations

import time
from typing import List, Dict, Any

from ingestion import load_directory, load_document, Document
from chunking import chunk_documents, Chunk
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from retrieval import Retriever
from generation import Generator


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 4,
        hybrid: bool = True,
        rerank: bool = True,
        alpha: float = 0.6,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.hybrid = hybrid
        self.rerank = rerank
        self.alpha = alpha

        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self.retriever = None
        self.generator = Generator()

        self.timings: Dict[str, float] = {}

    # ---------- Stage 1: ingestion ----------
    def ingest_directory(self, directory: str):
        t0 = time.time()
        self.documents.extend(load_directory(directory))
        self.timings["ingestion_sec"] = self.timings.get("ingestion_sec", 0) + (time.time() - t0)

    def ingest_file(self, path: str):
        t0 = time.time()
        self.documents.extend(load_document(path))
        self.timings["ingestion_sec"] = self.timings.get("ingestion_sec", 0) + (time.time() - t0)

    def ingest_texts(self, texts: List[str], source_prefix: str = "raw_text"):
        t0 = time.time()
        for i, text in enumerate(texts):
            self.documents.append(Document(text=text, source=f"{source_prefix}_{i}", metadata={"type": "raw"}))
        self.timings["ingestion_sec"] = self.timings.get("ingestion_sec", 0) + (time.time() - t0)

    # ---------- Stage 2-4: chunk, embed, index ----------
    def build_index(self):
        t0 = time.time()
        self.chunks = chunk_documents(self.documents, self.chunk_size, self.chunk_overlap)
        self.timings["chunking_sec"] = time.time() - t0

        t0 = time.time()
        corpus = [c.text for c in self.chunks]
        self.embedding_model.fit(corpus)  # no-op for neural backends
        vectors = self.embedding_model.encode(corpus)
        self.timings["embedding_sec"] = time.time() - t0

        t0 = time.time()
        self.vector_store = VectorStore(dim=vectors.shape[1])
        metadatas = [{"source": c.source, "chunk_index": c.metadata["chunk_index"], "text_preview": c.text[:80]} for c in self.chunks]
        self.vector_store.add(vectors, metadatas)
        self.timings["indexing_sec"] = time.time() - t0

        self.retriever = Retriever(self.embedding_model, self.vector_store, self.chunks)

    # ---------- Stage 5-7: query -> retrieve -> generate ----------
    def ask(self, question: str) -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("Call build_index() before ask().")

        t0 = time.time()
        retrieved = self.retriever.retrieve(
            question, top_k=self.top_k, hybrid=self.hybrid, rerank=self.rerank, alpha=self.alpha
        )
        retrieval_sec = time.time() - t0

        t0 = time.time()
        result = self.generator.generate(question, retrieved)
        generation_sec = time.time() - t0

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": retrieved,
            "retrieval_sec": retrieval_sec,
            "generation_sec": generation_sec,
        }

    def system_report(self) -> Dict[str, Any]:
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_backend": self.embedding_model.name,
            "embedding_dim": getattr(self.embedding_model, "dim", None),
            "vector_store_backend": self.vector_store.backend_type if self.vector_store else None,
            "generation_backend": self.generator.name,
            "hybrid_search": self.hybrid,
            "rerank_enabled": self.rerank,
            "alpha": self.alpha,
            "top_k": self.top_k,
            "timings": self.timings,
        }
