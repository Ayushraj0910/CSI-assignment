"""
chunking.py
===========
Text Chunking Module.

Breaks unstructured raw text into smaller, semantically coherent chunks
so that retrieval later on can isolate precise, relevant passages instead
of whole documents.

Strategy
--------
We implement a "recursive/paragraph-aware" splitter (similar in spirit to
LangChain's RecursiveCharacterTextSplitter) that:
  1. Cleans whitespace/control noise.
  2. Splits on paragraph boundaries first, then sentences, then hard
     character windows as a last resort -- so chunk borders land on
     natural language boundaries whenever possible.
  3. Applies a configurable overlap between consecutive chunks so context
     isn't lost at chunk edges (important for QA over long documents).

A `chunk_size` is measured in characters (simple, dependency-free, and
model-agnostic). Word/token-based sizing is also exposed for experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(paragraph: str) -> List[str]:
    # lightweight sentence splitter, no external NLP dependency required
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s for s in sentences if s]


def recursive_chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[str]:
    """
    Splits text into chunks of roughly `chunk_size` characters, preferring
    to break on paragraph -> sentence -> raw-character boundaries, with
    `chunk_overlap` characters repeated between consecutive chunks.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    units: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = (current + " " + unit).strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(unit) > chunk_size:
                # hard-wrap oversized single sentence/unit
                for i in range(0, len(unit), chunk_size - chunk_overlap):
                    chunks.append(unit[i:i + chunk_size])
                current = ""
            else:
                current = unit
    if current:
        chunks.append(current)

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        chunks = overlapped

    return chunks


def chunk_documents(
    documents,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """Chunk a list of ingestion.Document objects into Chunk objects."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        pieces = recursive_chunk_text(doc.text, chunk_size, chunk_overlap)
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    id=f"{doc.source}::chunk_{idx}",
                    text=piece,
                    source=doc.source,
                    metadata={**doc.metadata, "chunk_index": idx, "chunk_chars": len(piece)},
                )
            )
    return all_chunks


if __name__ == "__main__":
    sample = "Paragraph one. It has some sentences. " * 20 + "\n\n" + "Paragraph two here. " * 20
    for c in recursive_chunk_text(sample, chunk_size=200, chunk_overlap=40):
        print(len(c), repr(c[:60]))
