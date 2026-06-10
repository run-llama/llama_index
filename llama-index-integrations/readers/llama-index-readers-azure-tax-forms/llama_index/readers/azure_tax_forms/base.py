"""
AzureTaxFormReader — LlamaIndex BaseReader for IRS tax form extraction.

Extracts structured key-value pairs from IRS tax form PDFs using Azure
Document Intelligence. Supports Form 1040, W-2, Schedule C/E/K-1,
Form 1065, 1120, and 1120-S.

Usage::

    from llama_index.readers.azure_tax_forms import AzureTaxFormReader

    reader = AzureTaxFormReader(
        endpoint="https://my-resource.cognitiveservices.azure.com/",
        api_key="...",
    )

    # From file path
    docs = reader.load_data("path/to/1040.pdf")

    # From raw bytes (S3, blob storage, form upload, etc.)
    docs = reader.load_data_from_bytes("1040.pdf", pdf_bytes)

Each returned ``Document`` contains:
  - ``text``:     ``"key | value"`` lines, one per extracted KV pair
  - ``metadata``: form_type, stage, kv_count, timing, error info
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama_index.readers.azure_tax_forms.extractor import TaxFormExtractor, ExtractionConfig
from llama_index.readers.azure_tax_forms.models import ExtractionResult
from llama_index.readers.azure_tax_forms import audit

logger = logging.getLogger(__name__)


class AzureTaxFormReader(BaseReader):
    """
    LlamaIndex reader that extracts key-value pairs from IRS tax form PDFs
    using Azure Document Intelligence.

    Supports:
      - Form 1040, W-2, Schedule C/E/K-1
      - Form 1065, 1120, 1120-S
      - Any PDF processable by the ``prebuilt-document`` Azure DI model

    Features:
      - 4-stage recovery chain (direct → page-split → DPI-reduce → rotate)
      - Exponential back-off on Azure DI 429 rate-limit responses
        with ``Retry-After`` header support
      - Field normalisation (trailing spaces, quoted numerics, currency symbols)
      - Automatic form type inference from extracted key names
      - Per-document audit logging (no PII in stdout)

    Args:
        endpoint:                    Azure Document Intelligence endpoint URL.
        api_key:                     Azure DI API key.
        model_id:                    Azure DI model (default: ``prebuilt-document``).
        pages_per_chunk:             Pages per chunk in Stage 1 split recovery.
        poll_timeout_seconds:        Per-call Azure DI timeout.
        rate_limit_max_retries:      Maximum 429 retry attempts.
        rate_limit_initial_delay_ms: Back-off initial delay (ms).
        rate_limit_max_delay_ms:     Back-off ceiling (ms).
        enable_audit_log:            Write extraction audit to file (default: True).
        audit_log_dir:               Directory for the audit log file.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_id: str = "prebuilt-document",
        pages_per_chunk: int = 10,
        poll_timeout_seconds: int = 120,
        rate_limit_max_retries: int = 5,
        rate_limit_initial_delay_ms: int = 1_000,
        rate_limit_max_delay_ms: int = 32_000,
        enable_audit_log: bool = True,
        audit_log_dir: str = "logs",
    ) -> None:
        config = ExtractionConfig(
            endpoint=endpoint,
            api_key=api_key,
            model_id=model_id,
            pages_per_chunk=pages_per_chunk,
            poll_timeout_seconds=poll_timeout_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_initial_delay_ms=rate_limit_initial_delay_ms,
            rate_limit_max_delay_ms=rate_limit_max_delay_ms,
        )
        self._extractor = TaxFormExtractor(config)
        audit.configure_audit_logger(log_dir=audit_log_dir, enabled=enable_audit_log)

    def load_data(
        self,
        file: Union[str, Path],
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Load and extract KV pairs from a single PDF file.

        Args:
            file:       Path to the PDF file.
            extra_info: Optional metadata merged into the returned Document.

        Returns:
            List containing one :class:`llama_index.core.schema.Document`.
        """
        path = Path(file)
        try:
            pdf_bytes = path.read_bytes()
        except OSError as exc:
            logger.error("Could not read file %s: %s", path, exc)
            pdf_bytes = b""
        return self._run_extraction(str(path), pdf_bytes, extra_info or {})

    def load_data_from_bytes(
        self,
        document_id: str,
        pdf_bytes: bytes,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Load and extract KV pairs from in-memory PDF bytes.

        Useful when documents come from blob storage, S3, a form upload,
        or any source that provides raw bytes.

        Args:
            document_id: Identifier for the document (filename, UUID, S3 key, etc.).
            pdf_bytes:   Raw PDF bytes.
            extra_info:  Optional metadata merged into the returned Document.

        Returns:
            List containing one :class:`llama_index.core.schema.Document`.
        """
        return self._run_extraction(document_id, pdf_bytes, extra_info or {})

    def _run_extraction(
        self,
        document_id: str,
        pdf_bytes: bytes,
        extra_info: dict,
    ) -> list[Document]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self._extractor.extract(document_id, pdf_bytes)
                )
                result: ExtractionResult = future.result()
        else:
            result = asyncio.run(self._extractor.extract(document_id, pdf_bytes))

        audit.write_header()
        audit.record(result)
        return [_to_document(result, extra_info)]


def _to_document(result: ExtractionResult, extra_info: dict) -> Document:
    lines = [
        f"{entry.key} | {entry.value if entry.value else '(blank)'}"
        for entry in result.entries
    ]
    text = "\n".join(lines) if lines else "(no key-value pairs extracted)"
    metadata = {**result.as_dict(), **extra_info}
    return Document(text=text, metadata=metadata)
