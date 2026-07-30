"""
pdf_loader.py — Robust PDF text extraction that works across PDF types:

  1. Normal typed-text PDFs      -> direct text extraction (fast, accurate)
  2. Scanned/photographed PDFs   -> OCR fallback per page
  3. Handwritten notes (photos)  -> OCR fallback per page (best-effort; see
                                     the honesty note at the bottom of this
                                     file about handwriting OCR limits)

Strategy: for EVERY page, try native text extraction first. Only if that
page yields too little text do we rasterize that specific page to an image
and run Tesseract OCR on it. This means a 50-page PDF where only 3 pages are
scanned images doesn't pay the (much slower) OCR cost for the other 47.
"""
import logging
import os
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# If native extraction returns fewer than this many non-whitespace characters
# for a page, we treat it as "no usable text" and fall back to OCR.
MIN_CHARS_FOR_NATIVE_TEXT = int(os.environ.get("PDF_MIN_NATIVE_CHARS", "20"))

# DPI used when rasterizing a page for OCR. Higher = more accurate but slower
# and more memory. 200-300 is a reasonable range for OCR; push higher only
# if you're getting poor OCR results on small/dense text.
OCR_DPI = int(os.environ.get("PDF_OCR_DPI", "200"))


def _extract_native_text(page) -> str:
    try:
        return (page.extract_text() or "").strip()
    except Exception:
        logger.exception("Native text extraction failed on a page")
        return ""


def _ocr_page_image(pil_image) -> str:
    try:
        import pytesseract
        # psm 6 = "assume a single uniform block of text", a reasonable
        # general-purpose setting for document pages (vs. the default,
        # which assumes a full page layout with columns/etc. and can do
        # worse on dense or handwritten text).
        return pytesseract.image_to_string(pil_image, config="--psm 6").strip()
    except Exception:
        logger.exception("OCR failed on a page image")
        return ""


def _rasterize_page(file_path: str, page_number_1_indexed: int) -> Optional["object"]:
    """Render a single PDF page to a PIL image for OCR. Returns None on failure."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            file_path,
            dpi=OCR_DPI,
            first_page=page_number_1_indexed,
            last_page=page_number_1_indexed,
        )
        return images[0] if images else None
    except Exception:
        logger.exception(
            "Failed to rasterize page %d of %s for OCR (is poppler-utils installed?)",
            page_number_1_indexed, file_path,
        )
        return None


def load_pdf(file_path: str) -> list[Document]:
    name = os.path.basename(file_path)

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf is not installed; cannot read %s", name)
        return []

    try:
        reader = PdfReader(file_path)
    except Exception:
        logger.exception("Failed to open PDF: %s", name)
        return []

    documents: list[Document] = []
    num_pages = len(reader.pages)
    ocr_page_count = 0

    for i, page in enumerate(reader.pages):
        page_number = i + 1  # 1-indexed for humans

        text = _extract_native_text(page)

        if len(text.replace(" ", "").replace("\n", "")) < MIN_CHARS_FOR_NATIVE_TEXT:
            # Native extraction found little/nothing -> this page is very
            # likely a scanned image or a photo of handwriting. Fall back
            # to OCR for THIS page only.
            logger.info(
                "Page %d/%d of %s has little/no native text (%d chars) — running OCR",
                page_number, num_pages, name, len(text),
            )
            image = _rasterize_page(file_path, page_number)
            if image is not None:
                ocr_text = _ocr_page_image(image)
                if ocr_text:
                    text = ocr_text
                    ocr_page_count += 1

        if text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={"source": name, "page": page_number},
                )
            )
        else:
            logger.warning(
                "Page %d/%d of %s produced no text (native or OCR) — skipped",
                page_number, num_pages, name,
            )

    if ocr_page_count:
        logger.info(
            "%s: %d/%d pages required OCR fallback", name, ocr_page_count, num_pages
        )

    if not documents:
        logger.warning(
            "No text could be extracted from %s at all (native or OCR)", name
        )

    return documents


# ----------------------------------------------------------------------
# Honesty note, not code: Tesseract OCR (pytesseract) is trained mainly on
# printed/typed fonts. On genuinely cursive/messy handwriting it is often
# unreliable — expect noisy or partially wrong text on true handwritten
# pages even with this fallback working correctly. This loader guarantees
# that handwritten pages get READ (rather than silently producing nothing),
# but it can't guarantee the OCR text is fully accurate. If your handwritten
# notes are producing garbled/wrong text even after this fix, the loader is
# working as intended — the limitation is Tesseract's handwriting accuracy,
# and the real fix at that point is a cloud handwriting-OCR API (e.g. Google
# Cloud Vision's DOCUMENT_TEXT_DETECTION or Azure AI Vision's Read API),
# which are far more accurate on handwriting but require external API calls
# and (usually) an API key/cost.
# ----------------------------------------------------------------------
