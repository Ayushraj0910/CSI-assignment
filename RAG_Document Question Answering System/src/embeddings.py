"""
embeddings.py
=============
Embedding Module.

Maps chunked text strings into equivalent vector representations using a
pre-trained embedding model.

Backend selection
------------------
Primary  : `sentence-transformers` ("all-MiniLM-L6-v2", 384-dim) -- a strong,
           widely-used open embedding model. Used automatically when the
           package + model weights are available (requires internet on
           first run to download weights).
Fallback : scikit-learn `TfidfVectorizer` projected to a fixed-width dense
           vector via SVD (a classic, fully-offline, dependency-light
           "embedding"). This keeps the pipeline runnable in network-
           restricted / air-gapped environments while preserving the same
           `.encode(texts) -> np.ndarray` interface, so nothing else in the
           pipeline needs to change when swapping backends.

Both backends expose the same interface:
    model = EmbeddingModel()
    vectors = model.encode(["some text", "more text"])   # np.ndarray [N, dim]
    model.dim                                            # embedding dimension
    model.name                                           # human readable id
"""

from __future__ import annotations

import numpy as np
from typing import List


class SentenceTransformerBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


class TfidfSvdBackend:
    """
    Fully offline fallback embedding backend.

    TF-IDF gives a sparse bag-of-words representation; truncated SVD
    (latent semantic analysis) projects it into a fixed-size dense space
    so it behaves like a neural embedding for downstream cosine-similarity
    search. Must be `fit()` once on the corpus before `encode()`.
    """

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.dim = dim
        self.name = f"tfidf-svd-{dim}d (offline fallback)"
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, corpus: List[str]):
        effective_dim = min(self.dim, max(1, len(corpus) - 1))
        if effective_dim != self.dim:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=effective_dim, random_state=42)
            self.dim = effective_dim
        sparse = self._vectorizer.fit_transform(corpus)
        self._svd.fit(sparse)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdBackend must be fit() on a corpus before encode().")
        sparse = self._vectorizer.transform(texts)
        dense = self._svd.transform(sparse)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class EmbeddingModel:
    """
    Facade that prefers the neural embedding backend and transparently
    falls back to the offline TF-IDF+SVD backend if sentence-transformers
    (or its model weights / internet access) is unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 256):
        try:
            self.backend = SentenceTransformerBackend(model_name)
            self.backend_type = "sentence-transformers"
        except Exception as e:  # noqa: BLE001 - broad on purpose for env portability
            print(f"[embeddings] sentence-transformers unavailable ({e.__class__.__name__}); "
                  f"falling back to offline TF-IDF+SVD backend.")
            self.backend = TfidfSvdBackend(dim=fallback_dim)
            self.backend_type = "tfidf-svd"

        self.name = self.backend.name

    def fit(self, corpus: List[str]):
        """Only meaningful for the fallback backend; no-op for neural models."""
        if hasattr(self.backend, "fit"):
            self.backend.fit(corpus)
        self.dim = self.backend.dim

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.backend.encode(texts)


if __name__ == "__main__":
    model = EmbeddingModel()
    corpus = ["The cat sat on the mat.", "Dogs are loyal animals.", "Stock markets fell today."]
    model.fit(corpus)
    vecs = model.encode(corpus)
    print(model.name, "dim=", model.dim, "shape=", vecs.shape)
