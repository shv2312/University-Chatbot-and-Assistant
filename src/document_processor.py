"""Document ingestion and text extraction engine.

Supports PDF, DOCX, TXT, and Markdown files with text chunking
and metadata tracking for vector store indexing.
"""

import os
from typing import List, Dict, Any


def extract_text_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from supported document types (.pdf, .docx, .txt, .md).

    Args:
        file_path: Path to the target document.

    Returns:
        A list of dictionaries containing extracted text and page numbers:
        [{"text": page_or_doc_text, "page": page_number}]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_from_pdf(file_path)
    elif ext == ".docx":
        return _extract_from_docx(file_path)
    elif ext in [".txt", ".md"]:
        return _extract_from_text_or_md(file_path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt, .md"
        )


def _extract_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract page-by-page text from a PDF file using PyMuPDF (strictly 1-based page index)."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    pages = []
    doc = fitz.open(file_path)
    try:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1  # Strictly 1-based page index
            text = page.get_text()
            if text and text.strip():
                pages.append({"text": text.strip(), "page": page_num})
    finally:
        doc.close()
    return pages


def _extract_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract paragraph text from a Word document using python-docx."""
    import docx

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs).strip()

    if full_text:
        return [{"text": full_text, "page": 1}]
    return []


def _extract_from_text_or_md(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from plain text or Markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    if text:
        return [{"text": text, "page": 1}]
    return []


def chunk_text(
    extracted_pages: List[Dict[str, Any]],
    filename: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """Chunk extracted page texts while preserving metadata.

    Args:
        extracted_pages: List of dicts with 'text' and 'page' keys.
        filename: Original file name or path.
        chunk_size: Maximum characters per chunk (default 800).
        overlap: Character overlap between consecutive chunks (default 150).

    Returns:
        List of chunks with IDs, text, and metadata:
        [{"id": f"{filename}_p{page}_c{i}", "text": chunk_text, "metadata": {"filename": filename, "page": page}}]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    # Use the basename of the filename for clean identifier and metadata
    clean_filename = os.path.basename(filename) if filename else "document"

    chunks = []
    step = chunk_size - overlap

    for page_dict in extracted_pages:
        page_num = int(page_dict.get("page", 1))
        text = page_dict.get("text", "")
        if not text or not text.strip():
            continue

        clean_text = text.strip()
        text_len = len(clean_text)

        if text_len <= chunk_size:
            chunks.append(
                {
                    "id": f"{clean_filename}_p{page_num}_c0",
                    "text": clean_text,
                    "metadata": {
                        "filename": clean_filename,
                        "page": page_num,
                    },
                }
            )
        else:
            start = 0
            chunk_idx = 0
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_str = clean_text[start:end]
                chunks.append(
                    {
                        "id": f"{clean_filename}_p{page_num}_c{chunk_idx}",
                        "text": chunk_str,
                        "metadata": {
                            "filename": clean_filename,
                            "page": page_num,
                        },
                    }
                )
                chunk_idx += 1
                if end >= text_len:
                    break
                start += step

    return chunks
