"""
Core extraction engine — single-document extraction with 4-stage recovery chain.

Recovery chain:
  Stage 0  — direct Azure DI call on original bytes
  Stage 1  — page-split into chunks, analyse sequentially
  Stage 2  — DPI reduction to 300 DPI (rasterise)
  Stage 3  — rotation block: as-is → 90° → 180° → 270°

Exponential back-off with ±20% jitter on HTTP 429 responses.

Multi-document concurrent processing and the semaphore concurrency gate
are available in the enterprise edition.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from llama_index.readers.azure_tax_forms.models import KvEntry, ExtractionResult, TaxFormType
from llama_index.readers.azure_tax_forms.normalizer import normalize_pairs
from llama_index.readers.azure_tax_forms import pdf_utils

logger = logging.getLogger(__name__)

TARGET_DPI = 300


@dataclass
class ExtractionConfig:
    """Configuration for the extraction engine."""

    endpoint: str
    api_key: str
    model_id: str = "prebuilt-document"
    pages_per_chunk: int = 10
    poll_timeout_seconds: int = 120
    rate_limit_max_retries: int = 5
    rate_limit_initial_delay_ms: int = 1_000
    rate_limit_max_delay_ms: int = 32_000


class TaxFormExtractor:
    """
    Async extraction engine for IRS tax form documents.

    Instantiate once and reuse across extractions.

    Args:
        config: Extraction configuration.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key),
        )

    async def extract(self, document_id: str, pdf_bytes: bytes) -> ExtractionResult:
        """
        Run the full 4-stage extraction pipeline on ``pdf_bytes``.

        Args:
            document_id: Caller-supplied identifier (file path, UUID, etc.).
            pdf_bytes:   Raw PDF file bytes.

        Returns:
            :class:`ExtractionResult` — always returns, never raises.
        """
        start_ms = _now_ms()
        result = ExtractionResult(document_id=document_id)

        try:
            pairs, stage = await self._analyze_bytes(pdf_bytes, document_id)
            result.entries = [
                KvEntry(key=k, value=v, confidence=c) for k, v, c in pairs
            ]
            result.stage = stage
            result.form_type = _infer_form_type(result.entries)
        except Exception as exc:
            logger.error("Extraction failed for document_id=%s: %s", document_id, exc)
            result.error = str(exc)

        result.total_ms = _now_ms() - start_ms
        return result

    # ------------------------------------------------------------------
    # Recovery chain
    # ------------------------------------------------------------------

    async def _analyze_bytes(
        self,
        pdf_bytes: bytes,
        document_id: str,
    ) -> tuple[list[tuple[Optional[str], Optional[str], Optional[float]]], str]:
        """Run the 4-stage recovery chain. Returns (normalised_pairs, stage_label)."""

        # Stage 0: direct call
        try:
            raw = await self._analyze_once(pdf_bytes)
            if raw:
                return normalize_pairs(raw), "STAGE-0"
        except HttpResponseError as exc:
            if _is_quota_exhausted(exc):
                logger.error("Azure DI quota exhausted — upgrade to S0 paid tier")
                return [], "QUOTA-403"
            if not _is_oversize_error(exc):
                raise

        # Stage 1: page split (sequential)
        split_pairs = await self._analyze_split_chunks(pdf_bytes, document_id)
        if split_pairs:
            return normalize_pairs(split_pairs), "STAGE-1"

        # Stage 2: DPI reduction
        reduced = pdf_utils.reduce_dpi(pdf_bytes, TARGET_DPI)

        # Stage 3: rotation block
        rotation_pairs = await self._analyze_rotation_block(reduced, document_id)
        stage = "STAGE-2/3" if rotation_pairs else "STAGE-2/3-EMPTY"
        return normalize_pairs(rotation_pairs), stage

    async def _analyze_rotation_block(
        self,
        pdf_bytes: bytes,
        document_id: str,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """As-is → 90° → 180° → 270°. Short-circuits on first non-empty result."""
        try:
            raw = await self._analyze_once(pdf_bytes)
            if raw:
                return raw
        except HttpResponseError as exc:
            logger.warning("STAGE-3 as-is error for document_id=%s: %s", document_id, exc)

        for degrees in (90, 180, 270):
            try:
                rotated = pdf_utils.rotate_pdf(pdf_bytes, degrees)
                raw = await self._analyze_once(rotated)
                if raw:
                    return raw
            except Exception as exc:
                logger.warning("STAGE-3 %d° failed for document_id=%s: %s",
                               degrees, document_id, exc)
        return []

    async def _analyze_split_chunks(
        self,
        pdf_bytes: bytes,
        document_id: str,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """Split PDF into page chunks and analyse sequentially."""
        chunks = pdf_utils.split_by_page_count(pdf_bytes, self._config.pages_per_chunk)
        merged: list[tuple[Optional[str], Optional[str], Optional[float]]] = []
        for idx, chunk in enumerate(chunks):
            try:
                result = await self._analyze_once(chunk)
                merged.extend(result)
            except Exception as exc:
                logger.error("STAGE-1 chunk %d/%d failed for document_id=%s: %s",
                             idx + 1, len(chunks), document_id, exc)
        return merged

    # ------------------------------------------------------------------
    # Single Azure DI call with 429 retry back-off
    # ------------------------------------------------------------------

    async def _analyze_once(
        self,
        pdf_bytes: bytes,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """Single Azure DI call with exponential back-off on 429 responses."""
        delay_ms = float(self._config.rate_limit_initial_delay_ms)

        for attempt in range(self._config.rate_limit_max_retries + 1):
            try:
                return await self._do_analyze_once(pdf_bytes)
            except HttpResponseError as exc:
                if exc.status_code != 429:
                    raise
                if attempt == self._config.rate_limit_max_retries:
                    logger.warning("Azure DI 429 — exhausted %d retries",
                                   self._config.rate_limit_max_retries)
                    raise

                sleep_ms = _retry_after_ms(exc) or delay_ms
                jitter = sleep_ms * 0.2 * (random.random() * 2 - 1)
                actual_ms = max(1.0, min(sleep_ms + jitter, self._config.rate_limit_max_delay_ms))
                logger.info("Azure DI 429 — attempt %d/%d, sleeping %.0f ms",
                            attempt + 1, self._config.rate_limit_max_retries, actual_ms)
                await asyncio.sleep(actual_ms / 1_000)
                delay_ms = min(delay_ms * 2, self._config.rate_limit_max_delay_ms)

        raise RuntimeError("_analyze_once: unreachable")

    async def _do_analyze_once(self, pdf_bytes: bytes) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._call_azure_sync, pdf_bytes)

    def _call_azure_sync(self, pdf_bytes: bytes) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        poller = self._client.begin_analyze_document(self._config.model_id, pdf_bytes)
        result = poller.result()
        if not result.key_value_pairs:
            return []
        return [
            (
                kvp.key.content if kvp.key else None,
                kvp.value.content if kvp.value else None,
                kvp.confidence,
            )
            for kvp in result.key_value_pairs
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_oversize_error(exc: HttpResponseError) -> bool:
    return exc.status_code == 400 and "InvalidContentLength" in str(exc)


def _is_quota_exhausted(exc: HttpResponseError) -> bool:
    return exc.status_code == 403 and "Out of call volume quota" in str(exc)


def _retry_after_ms(exc: HttpResponseError) -> Optional[float]:
    try:
        header = exc.response.headers.get("Retry-After")
        if header:
            return float(header.strip()) * 1_000
    except Exception:
        pass
    return None


def _infer_form_type(entries: list[KvEntry]) -> TaxFormType:
    keys_lower = {e.key.lower() for e in entries if e.key}
    if any("1040" in k for k in keys_lower):
        return TaxFormType.FORM_1040
    if any("w-2" in k or "w2" in k or "employer" in k for k in keys_lower):
        return TaxFormType.W2
    if any("schedule c" in k or "profit or loss from business" in k for k in keys_lower):
        return TaxFormType.SCHEDULE_C
    if any("schedule e" in k or "supplemental income" in k for k in keys_lower):
        return TaxFormType.SCHEDULE_E
    if any("schedule k-1" in k or "partner's share" in k for k in keys_lower):
        return TaxFormType.SCHEDULE_K1
    if any("1065" in k for k in keys_lower):
        return TaxFormType.FORM_1065
    if any("1120-s" in k for k in keys_lower):
        return TaxFormType.FORM_1120S
    if any("1120" in k for k in keys_lower):
        return TaxFormType.FORM_1120
    return TaxFormType.UNKNOWN


def _now_ms() -> int:
    return int(time.monotonic() * 1_000)
