import io
from dataclasses import dataclass
from typing import List, Dict, Iterable
import fitz  # PyMuPDF


@dataclass
class PageText:
    page_num: int  # 1-based
    text: str


def extract_pdf_text(file_path: str) -> List[PageText]:
    """Extract plain text from a PDF, page by page, with 1-based page numbers."""
    pages: List[PageText] = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(PageText(page_num=i + 1, text=text))
    return pages


def _split_into_paragraphs(text: str) -> List[str]:
    # Normalize line breaks; split on double newlines as a simple paragraph heuristic
    parts = [p.strip() for p in text.replace("\r", "\n").split("\n\n")]
    return [p for p in parts if p]


def smart_chunk_pages(pages: List[PageText], chunk_chars: int = 1000, overlap: int = 150) -> List[Dict]:
    """
    Chunk pages into paragraph-based segments with rolling window.
    Returns list of dicts: { 'text', 'page', 'chunk_idx' }.
    """
    chunks: List[Dict] = []
    for pg in pages:
        paragraphs = _split_into_paragraphs(pg.text)
        if not paragraphs:
            continue
        # Build chunks up to chunk_chars
        buf: List[str] = []
        buf_len = 0
        cidx = 0
        for para in paragraphs:
            if buf_len + len(para) + 1 <= chunk_chars:
                buf.append(para)
                buf_len += len(para) + 1
            else:
                if buf:
                    chunks.append({"text": "\n\n".join(buf), "page": pg.page_num, "chunk_idx": cidx})
                    cidx += 1
                # Start new buffer with overlap from the end of previous
                # For simplicity, overlap by characters within the combined text
                combined = "\n\n".join(buf)[-overlap:] if buf else ""
                candidate = (combined + ("\n\n" if combined else "") + para).strip()
                if len(candidate) > chunk_chars:
                    # If single paragraph is huge, hard-split it
                    for start in range(0, len(para), chunk_chars - overlap):
                        part = para[start:start + (chunk_chars - overlap)]
                        chunks.append({"text": part, "page": pg.page_num, "chunk_idx": cidx})
                        cidx += 1
                    buf, buf_len = [], 0
                else:
                    buf = [candidate]
                    buf_len = len(candidate)

        if buf:
            chunks.append({"text": "\n\n".join(buf), "page": pg.page_num, "chunk_idx": cidx})
    return chunks
