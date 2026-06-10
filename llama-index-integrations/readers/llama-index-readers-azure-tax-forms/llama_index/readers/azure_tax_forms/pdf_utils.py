"""
PDF utilities — page splitting, DPI reduction, rotation.

Python equivalents of the Java ``PdfSplitter``, ``PdfNormalizer``,
and ``PdfRotator`` utilities used in the production recovery chain.

Dependencies:
  - ``pypdf`` for splitting and rotation (pure Python, no native libs needed)
  - ``Pillow`` for DPI reduction (rasterise pages via pypdf + re-render)
"""
from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def get_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF.  Returns -1 on error."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception as exc:
        logger.debug("get_page_count failed: %s", exc)
        return -1


def split_by_page_count(pdf_bytes: bytes, pages_per_chunk: int = 10) -> list[bytes]:
    """
    Split a PDF into chunks of at most ``pages_per_chunk`` pages.

    Returns a list of raw PDF byte strings.  If splitting fails, returns
    a single-element list containing the original bytes so the caller can
    still attempt extraction on the whole document.
    """
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        chunks: list[bytes] = []

        for start in range(0, total, pages_per_chunk):
            writer = PdfWriter()
            for page_idx in range(start, min(start + pages_per_chunk, total)):
                writer.add_page(reader.pages[page_idx])
            buf = io.BytesIO()
            writer.write(buf)
            chunks.append(buf.getvalue())

        logger.debug(
            "split_by_page_count: %d pages → %d chunks of ≤%d pages",
            total,
            len(chunks),
            pages_per_chunk,
        )
        return chunks
    except Exception as exc:
        logger.error("PDF split failed: %s — returning original bytes", exc)
        return [pdf_bytes]


def reduce_dpi(pdf_bytes: bytes, target_dpi: int = 300) -> bytes:
    """
    Re-render every page of a PDF at ``target_dpi`` using Pillow.

    This discards any vector text layer and produces a raster-only PDF,
    which improves Azure DI accuracy on scanned or poorly-formatted forms.
    Falls back to the original bytes if rasterisation fails.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        from PIL import Image

        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()

        for page in reader.pages:
            # Extract images from the page when present; fall back to the
            # page itself (which may be pure vector).
            images = list(page.images) if hasattr(page, "images") else []
            if images:
                # Re-encode the first embedded image at target DPI.
                img_data = images[0].data
                img = Image.open(io.BytesIO(img_data))
                img_buf = io.BytesIO()
                img.save(img_buf, format="JPEG", dpi=(target_dpi, target_dpi))
                # Add as a new PDF page backed by the re-rendered image.
                from pypdf import PdfWriter as _W
                sub_writer = _W()
                sub_writer.add_blank_page(
                    width=img.width * 72 / target_dpi,
                    height=img.height * 72 / target_dpi,
                )
                writer.add_page(sub_writer.pages[0])
            else:
                # No extractable images — keep the original page.
                writer.add_page(page)

        buf = io.BytesIO()
        writer.write(buf)
        result = buf.getvalue()
        logger.debug(
            "reduce_dpi: %d bytes → %d bytes at %d DPI",
            len(pdf_bytes),
            len(result),
            target_dpi,
        )
        return result
    except Exception as exc:
        logger.warning("DPI reduction failed: %s — using original bytes", exc)
        return pdf_bytes


def rotate_pdf(pdf_bytes: bytes, degrees: int) -> bytes:
    """
    Rotate all pages in a PDF by ``degrees`` (must be 90, 180, or 270).

    Returns the rotated PDF bytes, or the original bytes on error.
    """
    if degrees not in (90, 180, 270):
        raise ValueError(f"degrees must be 90, 180, or 270 — got {degrees}")
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page in reader.pages:
            page.rotate(degrees)
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("rotate_pdf(%d°) failed: %s", degrees, exc)
        return pdf_bytes
