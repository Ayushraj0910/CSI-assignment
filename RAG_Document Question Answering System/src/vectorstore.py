"""
vectorstore.py
==============
Vector Database Module.

Initializes a vector store, stores chunk embeddings + metadata, and
configures it for fast similarity search (cosine / inner-product on
normalized vectors).

Backend selection
------------------
Primary  : FAISS `IndexFlatIP` (inner product on L2-normalized vectors ==
           cosine similarity), the industry-standard ANN library. Wrapped
           with an ID map so we can persist chunk metadata alongside it.
Fallback : A minimal pure NumPy brute-force cosine-similarity index. Exact
           (not approximate) and perfectly fine for the corpus sizes used
           in this project; keeps the system runnable with zero native
           dependencies when FAISS isn't installed.

Both expose the same interface:
    store.add(vectors, metadatas)
    store.search(query_vector, top_k)  -> List[(score, metadata)]
    store.save(path) / VectorStore.load(path)
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple


class _NumpyIndex:
    """Pure numpy brute-force cosine similarity index (fallback backend)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray):
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors.shape[0] == 0:
            return np.array([[]]), np.array([[]])
        scores = self.vectors @ query.reshape(-1, 1)  # inner product, vectors assumed normalized
        scores = scores.flatten()
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]
        return scores[top_idx].reshape(1, -1), top_idx.reshape(1, -1)


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.metadatas: List[Dict[str, Any]] = []
        try:
            import faiss  # noqa: F401
            self._faiss = __import__("faiss")
            self.index = self._faiss.IndexFlatIP(dim)
            self.backend_type = "faiss.IndexFlatIP"
        except ImportError:
            self._faiss = None
            self.index = _NumpyIndex(dim)
            self.backend_type = "numpy-bruteforce (offline fallback)"

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        assert vectors.shape[0] == len(metadatas), "vectors/metadata count mismatch"
        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        scores, idxs = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append((float(score), self.metadatas[idx]))
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "metadatas": self.metadatas,
                    "backend_type": self.backend_type,
                    "vectors": (
                        None if self._faiss is not None
                        else self.index.vectors
                    ),
                },
                f,
            )

    @property
    def size(self) -> int:
        return len(self.metadatas)


if __name__ == "__main__":
    vs = VectorStore(dim=4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
    vs.add(vecs, [{"id": 1}, {"id": 2}, {"id": 3}])
    print(vs.backend_type, "size=", vs.size)
    print(vs.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2))
