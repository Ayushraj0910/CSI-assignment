"""
retrieval.py
============
Query Processing + Retrieval Module.

- Converts an incoming question into a query vector using the same
  embedding model used for the corpus (the "user input route").
- Queries the vector store to isolate the most contextually relevant
  chunks (dense/semantic retrieval).
- Optionally blends in a keyword (BM25-style) score for hybrid search,
  which helps on queries containing exact names, numbers, or jargon that
  dense embeddings can under-weight.
- Optionally re-ranks the fused candidate set with a lightweight
  cross-encoder-style relevance score (falls back to a lexical-overlap
  heuristic if no cross-encoder model is available offline).

This module is the "system optimizations" experimentation surface
mentioned in the assignment: `hybrid`, `rerank`, and `alpha` are all
toggleable knobs so results can be A/B compared (see metrics_report.md).
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Minimal, dependency-free BM25 implementation for keyword search."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_tokens = corpus_tokens
        self.doc_lens = [len(toks) for toks in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.df = Counter()
        for toks in corpus_tokens:
            for term in set(toks):
                self.df[term] += 1
        self.n_docs = len(corpus_tokens)
        self.doc_term_counts = [Counter(toks) for toks in corpus_tokens]

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        for i, counts in enumerate(self.doc_term_counts):
            dl = self.doc_lens[i] or 1
            s = 0.0
            for term in query_tokens:
                if term not in counts:
                    continue
                f = counts[term]
                idf = self._idf(term)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores[i] = s
        return scores


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Retriever:
    def __init__(self, embedding_model, vector_store, chunks):
        """
        chunks: the List[Chunk] in the same order they were embedded/added
                to the vector store (needed for BM25 keyword scoring).
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunks = chunks
        self._bm25 = BM25([_tokenize(c.text) for c in chunks])
        self._chunk_by_source_idx = {
            (c.source, c.metadata.get("chunk_index")): c for c in chunks
        }

    def embed_query(self, query: str):
        """User input route: question text -> query vector."""
        return self.embedding_model.encode([query])[0]

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        hybrid: bool = True,
        alpha: float = 0.6,
        rerank: bool = True,
        candidate_pool: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Returns top_k results: [{"text", "source", "score", "vector_score",
        "keyword_score", "metadata"}], sorted best-first.

        alpha: weight on vector-similarity vs keyword score when hybrid=True
               (alpha=1.0 -> pure vector, alpha=0.0 -> pure keyword).
        """
        query_vec = self.embed_query(query)
        pool = max(top_k, candidate_pool)
        vector_hits = self.vector_store.search(query_vec, top_k=pool)

        candidates = []
        for score, meta in vector_hits:
            key = (meta["source"], meta.get("chunk_index"))
            chunk = self._chunk_by_source_idx.get(key)
            if chunk is None:
                continue
            candidates.append({"chunk": chunk, "vector_score": score, "metadata": meta})

        if hybrid and candidates:
            query_tokens = _tokenize(query)
            bm25_scores_all = self._bm25.score(query_tokens)
            chunk_index_lookup = {id(c): i for i, c in enumerate(self.chunks)}
            for cand in candidates:
                idx = chunk_index_lookup[id(cand["chunk"])]
                cand["keyword_score"] = bm25_scores_all[idx]
        else:
            for cand in candidates:
                cand["keyword_score"] = 0.0

        norm_vec = _minmax_norm([c["vector_score"] for c in candidates])
        norm_kw = _minmax_norm([c["keyword_score"] for c in candidates])
        for cand, nv, nk in zip(candidates, norm_vec, norm_kw):
            cand["fused_score"] = alpha * nv + (1 - alpha) * nk if hybrid else nv

        candidates.sort(key=lambda c: c["fused_score"], reverse=True)

        if rerank:
            candidates = self._rerank(query, candidates)

        results = []
        for cand in candidates[:top_k]:
            results.append(
                {
                    "text": cand["chunk"].text,
                    "source": cand["chunk"].source,
                    "score": cand.get("rerank_score", cand["fused_score"]),
                    "vector_score": cand["vector_score"],
                    "keyword_score": cand["keyword_score"],
                    "metadata": cand["metadata"],
                }
            )
        return results

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-ranking layer. Tries a real cross-encoder if sentence-transformers
        is available; otherwise falls back to a lexical-overlap heuristic
        (query-term coverage ratio) blended with the fused retrieval score.
        This is deliberately a *second, independent* relevance signal so it
        can correct cases where vector similarity alone over- or under-rates
        a passage.
        """
        try:
            from sentence_transformers import CrossEncoder
            if not hasattr(self, "_cross_encoder"):
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, c["chunk"].text) for c in candidates]
            ce_scores = self._cross_encoder.predict(pairs)
            for c, s in zip(candidates, ce_scores):
                c["rerank_score"] = float(s)
        except Exception:
            query_terms = set(_tokenize(query))
            for c in candidates:
                chunk_terms = set(_tokenize(c["chunk"].text))
                overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
                c["rerank_score"] = 0.5 * c["fused_score"] + 0.5 * overlap

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates
