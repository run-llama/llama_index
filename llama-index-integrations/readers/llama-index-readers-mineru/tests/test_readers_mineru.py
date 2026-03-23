"""Tests for MinerUReader."""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.mineru import MinerUReader


class TestMinerUReaderClass:
    """Test class structure and MRO."""

    def test_class_inherits_base_reader(self):
        names_of_base_classes = [b.__name__ for b in MinerUReader.__mro__]
        assert BaseReader.__name__ in names_of_base_classes


class TestMinerUReaderValidation:
    """Test parameter validation."""

    @patch("llama_index.readers.mineru.base.MinerUReader._create_client")
    def test_invalid_mode_raises(self, mock_client):
        mock_client.return_value = MagicMock()
        with pytest.raises(ValueError, match="mode must be"):
            MinerUReader(mode="invalid")

    @patch("llama_index.readers.mineru.base.MinerUReader._create_client")
    def test_precision_without_token_raises(self, mock_client, monkeypatch):
        mock_client.return_value = MagicMock()
        monkeypatch.delenv("MINERU_TOKEN", raising=False)
        with pytest.raises(ValueError, match="precision mode requires a token"):
            MinerUReader(mode="precision", token=None)

    @patch("llama_index.readers.mineru.base.MinerUReader._create_client")
    def test_precision_with_env_token_ok(self, mock_client, monkeypatch):
        mock_client.return_value = MagicMock()
        monkeypatch.setenv("MINERU_TOKEN", "test-token-from-env")
        reader = MinerUReader(mode="precision")
        assert reader.mode == "precision"

    @patch("llama_index.readers.mineru.base.MinerUReader._create_client")
    def test_flash_mode_default(self, mock_client):
        mock_client.return_value = MagicMock()
        reader = MinerUReader()
        assert reader.mode == "flash"

    @patch("llama_index.readers.mineru.base.MinerUReader._create_client")
    def test_default_parameters(self, mock_client):
        mock_client.return_value = MagicMock()
        reader = MinerUReader()
        assert reader.language == "ch"
        assert reader.pages is None
        assert reader.timeout == 600
        assert reader.split_pages is False
        assert reader.ocr is False
        assert reader.formula is True
        assert reader.table is True


class TestMinerUReaderLoadData:
    """Test load_data with mocked MinerU client."""

    def _make_reader(self, **kwargs):
        with patch(
            "llama_index.readers.mineru.base.MinerUReader._create_client"
        ) as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            reader = MinerUReader(**kwargs)
        return reader, mock_client

    def _make_result(self, state="done", markdown="# Test\n\nHello world"):
        result = MagicMock()
        result.state = state
        result.markdown = markdown
        result.filename = "test.pdf"
        result.error = None
        return result

    def test_load_single_url(self):
        reader, client = self._make_reader()
        client.flash_extract.return_value = self._make_result()

        docs = reader.load_data("https://example.com/test.pdf")

        assert len(docs) == 1
        assert docs[0].text == "# Test\n\nHello world"
        assert docs[0].metadata["source"] == "https://example.com/test.pdf"
        assert docs[0].metadata["mode"] == "flash"
        client.flash_extract.assert_called_once()

    def test_load_multiple_sources(self):
        reader, client = self._make_reader()
        client.flash_extract.return_value = self._make_result()

        docs = reader.load_data(
            ["https://example.com/a.pdf", "https://example.com/b.pdf"]
        )

        assert len(docs) == 2
        assert client.flash_extract.call_count == 2

    def test_load_with_extra_info(self):
        reader, client = self._make_reader()
        client.flash_extract.return_value = self._make_result()

        docs = reader.load_data(
            "https://example.com/test.pdf",
            extra_info={"project": "demo"},
        )

        assert docs[0].metadata["project"] == "demo"
        assert docs[0].metadata["source"] == "https://example.com/test.pdf"

    def test_load_string_source(self):
        reader, client = self._make_reader()
        client.flash_extract.return_value = self._make_result()

        docs = reader.load_data("https://example.com/test.pdf")
        assert len(docs) == 1

    def test_precision_mode_calls_extract(self):
        with patch(
            "llama_index.readers.mineru.base.MinerUReader._create_client"
        ) as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            reader = MinerUReader(mode="precision", token="test-token")

        result = self._make_result()
        mock_client.extract.return_value = result

        docs = reader.load_data("https://example.com/test.pdf")

        assert len(docs) == 1
        mock_client.extract.assert_called_once()
        mock_client.flash_extract.assert_not_called()

    def test_failed_result_raises(self):
        reader, client = self._make_reader()
        failed_result = self._make_result(state="failed")
        failed_result.error = "file too large"
        client.flash_extract.return_value = failed_result

        with pytest.raises(ValueError, match="MinerU extraction failed"):
            reader.load_data("https://example.com/test.pdf")

    def test_metadata_contains_expected_keys(self):
        reader, client = self._make_reader(pages="1-5")
        client.flash_extract.return_value = self._make_result()

        docs = reader.load_data("https://example.com/test.pdf")
        meta = docs[0].metadata

        assert "source" in meta
        assert "mode" in meta
        assert "language" in meta
        assert "output_format" in meta
        assert meta["pages"] == "1-5"
        assert meta["output_format"] == "markdown"


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_is_url(self):
        from llama_index.readers.mineru.base import _is_url

        assert _is_url("https://example.com/file.pdf") is True
        assert _is_url("http://example.com/file.pdf") is True
        assert _is_url("/local/path/file.pdf") is False
        assert _is_url("file.pdf") is False

    def test_looks_like_pdf_by_extension(self):
        from llama_index.readers.mineru.base import _looks_like_pdf

        assert _looks_like_pdf("/path/to/file.pdf") is True
        assert _looks_like_pdf("/path/to/file.docx") is False
        assert _looks_like_pdf("https://example.com/file.pdf") is True
        assert _looks_like_pdf("https://example.com/file.pdf?token=abc") is True

    def test_parse_page_range_single(self):
        from llama_index.readers.mineru.base import _parse_page_range

        assert _parse_page_range("3") == {3}

    def test_parse_page_range_range(self):
        from llama_index.readers.mineru.base import _parse_page_range

        assert _parse_page_range("1-5") == {1, 2, 3, 4, 5}
