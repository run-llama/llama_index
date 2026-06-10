"""
Unit tests for TaxFormExtractor — Azure DI calls are fully mocked.
No real network calls, no real PDFs required.
"""
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError

from llama_index.readers.azure_tax_forms.extractor import (
    TaxFormExtractor,
    ExtractionConfig,
    _is_oversize_error,
    _is_quota_exhausted,
    _retry_after_ms,
)
from llama_index.readers.azure_tax_forms.models import TaxFormType


def _make_config(**overrides) -> ExtractionConfig:
    defaults = dict(
        endpoint="https://fake.cognitiveservices.azure.com/",
        api_key="fake-key",
        model_id="prebuilt-document",
        rate_limit_max_retries=2,
        rate_limit_initial_delay_ms=10,
        rate_limit_max_delay_ms=100,
    )
    defaults.update(overrides)
    return ExtractionConfig(**defaults)


def _make_extractor(**config_overrides) -> TaxFormExtractor:
    return TaxFormExtractor(_make_config(**config_overrides))


def _http_error(status: int, message: str = "") -> HttpResponseError:
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    err = HttpResponseError(message=message, response=response)
    err.status_code = status
    return err


class TestErrorClassification:
    def test_oversize_error_detected(self):
        assert _is_oversize_error(_http_error(400, "InvalidContentLength: too large"))

    def test_non_oversize_400(self):
        assert not _is_oversize_error(_http_error(400, "Some other error"))

    def test_quota_exhausted(self):
        assert _is_quota_exhausted(_http_error(403, "Out of call volume quota for this month"))

    def test_auth_403_not_quota(self):
        assert not _is_quota_exhausted(_http_error(403, "Access denied"))

    def test_retry_after_parsed(self):
        exc = _http_error(429)
        exc.response.headers = {"Retry-After": "5"}
        assert _retry_after_ms(exc) == 5_000.0

    def test_retry_after_missing(self):
        exc = _http_error(429)
        exc.response.headers = {}
        assert _retry_after_ms(exc) is None


class TestTaxFormExtractor:
    @pytest.mark.asyncio
    async def test_extract_stage0_success(self):
        extractor = _make_extractor()
        fake_pairs = [("Adjusted gross income", "75000", 0.99)]
        with patch.object(extractor, "_call_azure_sync", return_value=fake_pairs):
            result = await extractor.extract("test-doc", b"%PDF-fake")
        assert not result.is_empty
        assert result.stage == "STAGE-0"
        assert result.entries[0].key == "Adjusted gross income"

    @pytest.mark.asyncio
    async def test_extract_empty_on_failure(self):
        extractor = _make_extractor()
        with patch.object(extractor, "_call_azure_sync", return_value=[]):
            result = await extractor.extract("test-doc", b"%PDF-fake")
        assert result.is_empty

    @pytest.mark.asyncio
    async def test_429_retried_then_succeeds(self):
        extractor = _make_extractor(rate_limit_max_retries=2, rate_limit_initial_delay_ms=1)
        call_count = 0
        fake_pairs = [("Total income", "50000", 0.95)]

        def side_effect(pdf_bytes):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _http_error(429, "Too Many Requests")
            return fake_pairs

        with patch.object(extractor, "_call_azure_sync", side_effect=side_effect):
            result = await extractor.extract("test-doc", b"%PDF-fake")
        assert not result.is_empty
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_500_returns_empty_with_error(self):
        extractor = _make_extractor()
        with patch.object(extractor, "_call_azure_sync",
                          side_effect=_http_error(500, "Internal Server Error")):
            result = await extractor.extract("test-doc", b"%PDF-fake")
        assert result.is_empty
        assert result.error is not None


class TestFormTypeInference:
    @pytest.mark.asyncio
    async def test_infers_1040(self):
        extractor = _make_extractor()
        with patch.object(extractor, "_call_azure_sync",
                          return_value=[("Form 1040 — adjusted gross income", "75000", 0.9)]):
            result = await extractor.extract("1040.pdf", b"%PDF")
        assert result.form_type == TaxFormType.FORM_1040

    @pytest.mark.asyncio
    async def test_infers_w2(self):
        extractor = _make_extractor()
        with patch.object(extractor, "_call_azure_sync",
                          return_value=[("Employer identification number (W-2)", "12-3456789", 0.95)]):
            result = await extractor.extract("w2.pdf", b"%PDF")
        assert result.form_type == TaxFormType.W2

    @pytest.mark.asyncio
    async def test_unknown(self):
        extractor = _make_extractor()
        with patch.object(extractor, "_call_azure_sync",
                          return_value=[("Some generic field", "value", 0.8)]):
            result = await extractor.extract("unknown.pdf", b"%PDF")
        assert result.form_type == TaxFormType.UNKNOWN
