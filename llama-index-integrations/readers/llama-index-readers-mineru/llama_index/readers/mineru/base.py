"""
MinerU Reader — parse documents via MinerU API into llama_index Documents.

Supports two parsing modes:
- flash (default): Fast, no token required, Markdown-only output.
- precision: Full-featured, token required. Supports OCR, formula/table recognition.

Requires: pip install mineru-open-sdk pypdf
"""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

_SOURCE_TAG = "llama-index-mineru"


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _looks_like_pdf(value: str) -> bool:
    """Check whether *value* (path or URL) appears to be a PDF."""
    parsed = urlparse(value)
    path_suffix = PurePosixPath(parsed.path).suffix.lower()
    if path_suffix == ".pdf":
        return True
    if parsed.scheme not in {"http", "https"}:
        return False
    try:
        from urllib.request import Request, urlopen

        req = Request(value, method="HEAD")
        with urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            return "application/pdf" in content_type.lower()
    except Exception:
        logger.debug("HEAD request failed for %s, assuming not PDF", value)
        return False


def _parse_page_range(page_range: str) -> set[int]:
    """Parse a page range string like '1-5' or '3' into a set of 1-based page numbers."""
    page_range = page_range.strip()
    if "-" in page_range:
        start_s, end_s = page_range.split("-", 1)
        return set(range(int(start_s), int(end_s) + 1))
    return {int(page_range)}


def _split_pdf_to_pages(
    pdf_path: Path,
    page_numbers: set[int] | None = None,
) -> tuple[TemporaryDirectory, list[tuple[int, Path]]]:
    """Split a PDF into one-page temporary PDF files."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    temp_dir = TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    page_files: list[tuple[int, Path]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        if page_numbers is not None and page_number not in page_numbers:
            continue
        writer = PdfWriter()
        writer.add_page(page)
        page_path = temp_root / f"{pdf_path.stem}_page_{page_number}.pdf"
        with page_path.open("wb") as f:
            writer.write(f)
        page_files.append((page_number, page_path))

    return temp_dir, page_files


def _download_url_to_temp(url: str) -> tuple[TemporaryDirectory, Path]:
    """Download a URL to a temporary local PDF file."""
    from urllib.request import urlopen

    temp_dir = TemporaryDirectory()
    pdf_path = Path(temp_dir.name) / "downloaded.pdf"
    with urlopen(url) as response:
        pdf_path.write_bytes(response.read())
    return temp_dir, pdf_path


class MinerUReader(BaseReader):
    """
    Read documents using MinerU API and return llama_index Documents.

    Supports two parsing modes controlled by the ``mode`` parameter:

    - **flash** (default): Uses the MinerU Agent lightweight API. No token
      required, optimised for speed, Markdown-only output. Max 10 MB / 20 pages.
    - **precision**: Uses the MinerU standard extraction API. Requires a token
      (pass ``token`` or set ``MINERU_TOKEN`` env var). Supports OCR, formula
      and table recognition, multiple model versions. Max 200 MB / 600 pages.

    Both modes return Documents whose ``text`` is Markdown.

    Args:
        mode: ``"flash"`` (default) or ``"precision"``.
        token: MinerU API token for precision mode. Falls back to the
            ``MINERU_TOKEN`` environment variable.
        language: Document language code, default ``"ch"``.
        pages: Page range string, e.g. ``"1-10"``.
        timeout: Max seconds to wait for task completion, default ``600``.
        split_pages: If ``True``, split each PDF into one-page chunks and
            yield one Document per page.
        ocr: Enable OCR (precision mode only).
        formula: Enable formula recognition (precision mode only).
        table: Enable table recognition (precision mode only).

    """

    def __init__(
        self,
        mode: str = "flash",
        token: str | None = None,
        language: str = "ch",
        pages: str | None = None,
        timeout: int = 600,
        split_pages: bool = False,
        ocr: bool = False,
        formula: bool = True,
        table: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.token = token
        self.language = language
        self.pages = pages
        self.timeout = timeout
        self.split_pages = split_pages
        self.ocr = ocr
        self.formula = formula
        self.table = table

        self._validate()
        self._client = self._create_client()

    def _validate(self) -> None:
        if self.mode not in {"flash", "precision"}:
            raise ValueError("mode must be 'flash' or 'precision'")
        if self.mode == "precision" and not (
            self.token or os.environ.get("MINERU_TOKEN")
        ):
            raise ValueError(
                "precision mode requires a token. "
                "Pass token=... or set the MINERU_TOKEN environment variable."
            )

    def _create_client(self) -> Any:
        try:
            from mineru import MinerU
        except ImportError as exc:
            raise ImportError(
                "mineru-open-sdk is required to use MinerUReader. "
                "Install with: pip install mineru-open-sdk"
            ) from exc

        client = MinerU(token=self.token)
        client.set_source(_SOURCE_TAG)
        return client

    def load_data(
        self,
        sources: str | Path | list[str | Path],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Parse file paths or URLs and return Documents with Markdown content.

        Args:
            sources: A single file path / URL, or a list of them.
            extra_info: Optional metadata dict merged into every Document.

        Returns:
            List of Documents. Each Document's ``text`` is Markdown.

        """
        if isinstance(sources, (str, Path)):
            sources = [sources]
        sources = [str(s) for s in sources]

        documents: List[Document] = []
        for src in sources:
            documents.extend(self._load_single(src, extra_info))
        return documents

    def _load_single(self, src: str, extra_info: Optional[Dict]) -> List[Document]:
        if self.split_pages and _looks_like_pdf(src):
            return self._load_split_pdf(src, extra_info)
        return [self._load_whole(src, extra_info)]

    def _load_whole(
        self,
        src: str,
        extra_info: Optional[Dict],
        page: int | None = None,
    ) -> Document:
        result = self._extract(src)
        self._check_result(result, src, page)

        metadata = self._build_metadata(src, result, page)
        if extra_info:
            metadata.update(extra_info)

        return Document(text=result.markdown, metadata=metadata)

    def _load_split_pdf(self, src: str, extra_info: Optional[Dict]) -> List[Document]:
        download_tmp: TemporaryDirectory | None = None
        split_tmp: TemporaryDirectory | None = None
        documents: List[Document] = []

        try:
            if _is_url(src):
                download_tmp, local_path = _download_url_to_temp(src)
            else:
                local_path = Path(src)
                if not local_path.exists():
                    raise FileNotFoundError(f"PDF not found: {src}")

            target_pages = _parse_page_range(self.pages) if self.pages else None
            split_tmp, page_files = _split_pdf_to_pages(local_path, target_pages)

            for page_number, page_path in page_files:
                result = self._extract(str(page_path), use_page_range=False)
                self._check_result(result, src, page_number)

                metadata = self._build_metadata(src, result, page_number)
                if extra_info:
                    metadata.update(extra_info)

                documents.append(Document(text=result.markdown, metadata=metadata))
        finally:
            if split_tmp is not None:
                split_tmp.cleanup()
            if download_tmp is not None:
                download_tmp.cleanup()

        return documents

    def _extract(self, src: str, use_page_range: bool = True) -> Any:
        """Call the MinerU SDK to parse a single source."""
        kwargs: Dict[str, Any] = {
            "language": self.language,
            "timeout": self.timeout,
        }

        if self.mode == "flash":
            if use_page_range and self.pages:
                kwargs["page_range"] = self.pages
            return self._client.flash_extract(src, **kwargs)

        kwargs.update(
            {
                "ocr": self.ocr,
                "formula": self.formula,
                "table": self.table,
            }
        )
        if use_page_range and self.pages:
            kwargs["pages"] = self.pages
        return self._client.extract(src, **kwargs)

    def _build_metadata(
        self,
        source: str,
        result: Any,
        page: int | None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source": source,
            "mode": self.mode,
            "language": self.language,
            "output_format": "markdown",
            "filename": getattr(result, "filename", None),
        }
        if page is not None:
            metadata["page"] = page
        if self.pages:
            metadata["pages"] = self.pages
        return metadata

    @staticmethod
    def _check_result(result: Any, source: str, page: int | None) -> None:
        state = getattr(result, "state", None)
        if state == "done":
            return
        error = getattr(result, "error", None)
        loc = f", page={page}" if page is not None else ""
        raise ValueError(
            f"MinerU extraction failed: source={source}{loc}, "
            f"state={state}, error={error}"
        )
