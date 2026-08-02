"""
generation.py
=============
Answer Generation Module.

Connects retrieved context chunks + the original query into a single
prompt, sends it to a language model, and returns a grounded answer with
source citations.

Backend selection
------------------
Primary  : Any hosted chat-completion LLM (OpenAI, Anthropic, etc.) via a
           thin adapter -- plug in an API key and this module will use it
           automatically. This is the recommended production backend.
Fallback : `LocalExtractiveGenerator`, a fully offline generator that
           builds a grounded answer directly from the retrieved chunks
           (sentence selection ranked by query-term overlap) when no LLM
           API key / local model is available. It cannot "hallucinate"
           since it never emits text outside the retrieved context --
           useful both as an offline demo and as a sanity baseline to
           compare an LLM's answers against.

Both backends share the interface:
    generator.generate(query, retrieved_chunks) -> {"answer": str, "sources": [...]}
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Any


PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer that."
Cite the source file name(s) you used.

Context:
{context}

Question: {question}

Answer (grounded strictly in the context above):"""


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, r in enumerate(retrieved_chunks):
        context_blocks.append(f"[{i+1}] (source: {r['source']})\n{r['text']}")
    context = "\n\n".join(context_blocks)
    return PROMPT_TEMPLATE.format(context=context, question=query)


class OpenAIChatGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.name = f"openai/{model}"

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = build_prompt(query, retrieved_chunks)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer, "sources": [r["source"] for r in retrieved_chunks], "prompt": prompt}


class AnthropicChatGenerator:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic/{model}"

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = build_prompt(query, retrieved_chunks)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text
        return {"answer": answer, "sources": [r["source"] for r in retrieved_chunks], "prompt": prompt}


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "which", "who", "whom", "how", "why", "when", "where",
    "does", "do", "did", "has", "have", "had", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "as", "by", "with", "at", "from",
    "into", "than", "then", "so", "such", "can", "could", "would", "should",
    "will", "shall", "may", "might", "not", "no", "but", "if", "each", "their",
    "his", "her", "he", "she", "they", "them", "you", "your", "i", "we", "our",
}


class LocalExtractiveGenerator:
    """
    Fully offline, hallucination-free fallback: composes a grounded answer
    by selecting and lightly stitching together the most query-relevant
    sentences from the retrieved chunks. No external API or model download
    required, which is why this is the default in this sandboxed demo run.

    Crucially, it never invents an answer: if no retrieved sentence has
    meaningful lexical overlap with the (stopword-filtered) question terms,
    it explicitly reports that the documents don't contain the answer,
    rather than falling back to an arbitrary excerpt.
    """

    MIN_TERM_LEN = 3
    MIN_OVERLAP = 1

    def __init__(self):
        self.name = "local-extractive (offline fallback, no external LLM)"

    def _split_sentences(self, text: str) -> List[str]:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sents if s.strip()]

    def _content_terms(self, text: str) -> set:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {t for t in tokens if t not in _STOPWORDS and len(t) >= self.MIN_TERM_LEN}

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = build_prompt(query, retrieved_chunks)
        no_info_answer = "I don't have enough information in the provided documents to answer that."

        if not retrieved_chunks:
            return {"answer": no_info_answer, "sources": [], "prompt": prompt}

        query_terms = self._content_terms(query)
        scored_sentences = []
        for r in retrieved_chunks:
            for sent in self._split_sentences(r["text"]):
                sent_terms = self._content_terms(sent)
                overlap = len(query_terms & sent_terms)
                if overlap >= self.MIN_OVERLAP:
                    scored_sentences.append((overlap, sent, r["source"]))

        if not scored_sentences:
            return {"answer": no_info_answer, "sources": [], "prompt": prompt}

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best = scored_sentences[:3]
        answer = " ".join(s for _, s, _ in best)
        sources = list(dict.fromkeys(src for _, _, src in best))

        answer = f"{answer}\n\n(Grounded in: {', '.join(sources)})"
        return {"answer": answer, "sources": sources, "prompt": prompt}


class Generator:
    """
    Facade: tries Anthropic, then OpenAI (if API keys are configured),
    and otherwise transparently uses the offline extractive fallback so
    the pipeline always runs end-to-end.
    """

    def __init__(self, preferred: str = "auto"):
        self.backend = None
        order = [preferred] if preferred != "auto" else ["anthropic", "openai", "local"]
        for choice in order:
            try:
                if choice == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                    self.backend = AnthropicChatGenerator()
                    break
                if choice == "openai" and os.environ.get("OPENAI_API_KEY"):
                    self.backend = OpenAIChatGenerator()
                    break
                if choice == "local":
                    self.backend = LocalExtractiveGenerator()
                    break
            except Exception:
                continue
        if self.backend is None:
            self.backend = LocalExtractiveGenerator()

        self.name = self.backend.name

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.backend.generate(query, retrieved_chunks)
