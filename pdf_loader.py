"""
pdf_loader.py — Extracts text from PDF files.

Tries the embedded text layer first (fast), and falls back to OCR
(pdf2image + pytesseract) page-by-page for scanned documents.
"""

import logging
import os

from langchain_core.documents import Document
from pdf2image import convert_from_path
from pypdf import PdfReader
import pytesseract

logger = logging.getLogger(__name__)

OCR_DPI = int(os.environ.get("OCR_DPI", "200"))


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF, returning one Document per page that has extractable text."""
    documents: list[Document] = []
    pdf_name = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
    except Exception:
        logger.exception("Could not open PDF: %s", pdf_name)
        return documents

    for page_number, page in enumerate(reader.pages, start=1):
        text = ""

        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            logger.warning(
                "Embedded text extraction failed on page %s of %s",
                page_number, pdf_name,
            )

        if not text:
            text = _ocr_page(file_path, page_number, pdf_name)

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_name, "page": page_number},
                )
            )

    return documents


def _ocr_page(file_path: str, page_number: int, pdf_name: str) -> str:
    """OCR fallback for a single scanned page (no embedded text layer)."""
    try:
        images = convert_from_path(
            file_path, dpi=OCR_DPI, first_page=page_number, last_page=page_number
        )
    except Exception:
        logger.warning(
            "OCR page-render failed on page %s of %s (is poppler-utils installed?)",
            page_number, pdf_name,
        )
        return ""

    if not images:
        return ""

    try:
        text = pytesseract.image_to_string(images[0], config="--psm 6").strip()
    except Exception:
        logger.warning(
            "Tesseract OCR failed on page %s of %s (is tesseract-ocr installed?)",
            page_number, pdf_name,
        )
        text = ""
    finally:
        images[0].close()

    return text
