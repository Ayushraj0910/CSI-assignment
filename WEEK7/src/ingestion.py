"""
ingestion.py
============
Document Ingestion Module.

Accepts custom text inputs: raw .txt/.md notes, PDF files, or domain-specific
Hugging Face dataset archives, and normalizes them all into a common
`Document` representation: {"text": str, "source": str, "metadata": dict}.

Design notes
------------
- Every loader returns a list[Document] so multi-file / multi-record sources
  (e.g. a HF dataset with many rows) plug into the rest of the pipeline
  without any special-casing downstream.
- PDF parsing uses `pypdf` (pure python, no native deps) which is available
  in this environment. Hugging Face dataset loading uses the `datasets`
  library when installed; if it isn't, we raise a clear, actionable error
  instead of failing silently.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Document:
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _read_txt(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(text=text, source=os.path.basename(path), metadata={"type": "text"})]


def _read_pdf(path: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to read PDFs. Install with `pip install pypdf`."
        ) from e

    reader = PdfReader(path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n".join(pages_text)
    return [
        Document(
            text=full_text,
            source=os.path.basename(path),
            metadata={"type": "pdf", "num_pages": len(reader.pages)},
        )
    ]


def _read_hf_dataset(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    max_records: int = 200,
) -> List[Document]:
    """
    Loads a Hugging Face dataset (e.g. 'vectara/open_ragbench') and converts
    each record's text field into a Document.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The `datasets` library is required for Hugging Face sources. "
            "Install with `pip install datasets`."
        ) from e

    ds = load_dataset(dataset_name, split=split)
    docs: List[Document] = []
    for i, row in enumerate(ds):
        if i >= max_records:
            break
        text = row.get(text_field, "")
        if not text:
            continue
        docs.append(
            Document(
                text=text,
                source=f"{dataset_name}::{split}::row_{i}",
                metadata={"type": "hf_dataset", "dataset": dataset_name, "row": i},
            )
        )
    return docs


def load_document(path: str) -> List[Document]:
    """Dispatch a single file path to the right loader based on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def load_directory(directory: str, patterns=("*.txt", "*.md", "*.pdf")) -> List[Document]:
    """Load every supported file inside a directory (non-recursive)."""
    docs: List[Document] = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            docs.extend(load_document(path))
    return docs


def load_hf_source(dataset_name: str, **kwargs) -> List[Document]:
    """Public entrypoint for Hugging Face dataset ingestion."""
    return _read_hf_dataset(dataset_name, **kwargs)


if __name__ == "__main__":
    # quick manual smoke test
    import sys
    for p in sys.argv[1:]:
        for d in load_document(p):
            print(d.source, "->", len(d.text), "chars")
