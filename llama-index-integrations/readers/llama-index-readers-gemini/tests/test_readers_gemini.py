"""
Test suite for Gemini PDF Reader components.

This module contains tests for the GeminiReader class,
CacheManager and utility functions.
"""

import os
import time
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.gemini import GeminiReader
from llama_index.readers.gemini.types import ProcessingStats
from llama_index.readers.gemini.utils import is_web_url, download_from_url
from llama_index.readers.gemini.cache import CacheManager


class TestGeminiReader:
    """Test suite for GeminiReader class."""

    def test_inheritance(self):
        """Test that GeminiReader inherits from BaseReader."""
        names_of_base_classes = [b.__name__ for b in GeminiReader.__mro__]
        assert BaseReader.__name__ in names_of_base_classes


class TestCacheManager:
    """Test suite for CacheManager class."""

    @pytest.fixture()
    def setup_cache_dir(self):
        """Create a temporary directory for cache tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after tests
        shutil.rmtree(temp_dir)

    @pytest.fixture()
    def cache_manager(self, setup_cache_dir):
        """Create a CacheManager instance for testing."""
        return CacheManager(
            enable_caching=True,
            cache_dir=setup_cache_dir,
            cache_ttl=3600,  # 1 hour TTL
            verbose=False,
        )

    @pytest.fixture()
    def mock_documents(self):
        """Create sample documents for testing."""
        return [
            Document(text="Test document 1", metadata={"page": 1}),
            Document(text="Test document 2", metadata={"page": 2}),
        ]

    @pytest.fixture()
    def mock_stats(self):
        """Create sample processing stats for testing."""
        return ProcessingStats(
            start_time=time.time(),
            end_time=None,
            total_pages=2,
            processed_pages=2,
            total_chunks=5,
            errors=[],
        )

    def test_initialization(self, setup_cache_dir):
        """Test initialization of CacheManager."""
        # Test with caching enabled
        cache_manager = CacheManager(
            enable_caching=True,
            cache_dir=setup_cache_dir,
            cache_ttl=3600,
            verbose=False,
        )
        assert cache_manager.enable_caching is True
        assert cache_manager.cache_dir == setup_cache_dir
        assert cache_manager.cache_ttl == 3600
        assert os.path.exists(setup_cache_dir)

        # Test with caching disabled
        cache_manager_disabled = CacheManager(
            enable_caching=False,
            cache_dir=setup_cache_dir,
            cache_ttl=3600,
            verbose=False,
        )
        assert cache_manager_disabled.enable_caching is False
        assert len(cache_manager_disabled._cache) == 0

    def test_compute_file_hash(self, cache_manager, setup_cache_dir):
        """Test hash computation for files."""
        # Create a test file
        test_file_path = os.path.join(setup_cache_dir, "test_file.txt")
        with open(test_file_path, "w") as f:
            f.write("Test content")

        # Compute hash
        file_hash = cache_manager.compute_file_hash(test_file_path)

        # Verify hash is a non-empty string
        assert isinstance(file_hash, str)
        assert len(file_hash) > 0

        # Compute hash again and verify it's the same
        file_hash2 = cache_manager.compute_file_hash(test_file_path)
        assert file_hash == file_hash2

        # Modify file and verify hash changes
        time.sleep(0.1)  # Ensure modification time changes
        with open(test_file_path, "w") as f:
            f.write("Modified content")

        file_hash3 = cache_manager.compute_file_hash(test_file_path)
        assert file_hash != file_hash3

    def test_save_and_load_cache(self, cache_manager, mock_documents, mock_stats):
        """Test saving to cache and loading from cache."""
        file_hash = "test_hash_123"

        # Save documents to cache
        cache_manager.save_to_cache(file_hash, mock_documents, mock_stats)

        # Verify cache file was created
        cache_file_path = os.path.join(cache_manager.cache_dir, f"{file_hash}.json")
        assert os.path.exists(cache_file_path)

        # Load from cache
        loaded_documents = cache_manager.load_from_cache(file_hash)

        # Verify loaded documents match original
        assert len(loaded_documents) == len(mock_documents)
        for i, doc in enumerate(loaded_documents):
            assert doc.text == mock_documents[i].text
            assert doc.metadata == mock_documents[i].metadata

    def test_cache_expiration(self, setup_cache_dir, mock_documents, mock_stats):
        """Test cache expiration based on TTL."""
        # Create cache manager with short TTL
        short_ttl_cache = CacheManager(
            enable_caching=True,
            cache_dir=setup_cache_dir,
            cache_ttl=1,  # 1 second TTL
            verbose=False,
        )

        file_hash = "expiring_hash"

        # Save to cache
        short_ttl_cache.save_to_cache(file_hash, mock_documents, mock_stats)

        # Verify it's initially in cache
        assert short_ttl_cache.load_from_cache(file_hash) is not None

        # Wait for TTL to expire
        time.sleep(1.5)

        # Verify it's no longer in cache
        assert short_ttl_cache.load_from_cache(file_hash) is None

    def test_disabled_cache(self, setup_cache_dir, mock_documents, mock_stats):
        """Test behavior when caching is disabled."""
        # Create cache manager with caching disabled
        disabled_cache = CacheManager(
            enable_caching=False,
            cache_dir=setup_cache_dir,
            cache_ttl=3600,
            verbose=False,
        )

        file_hash = "test_disabled"

        # Attempt to save to cache
        disabled_cache.save_to_cache(file_hash, mock_documents, mock_stats)

        # Verify no cache file was created
        cache_file_path = os.path.join(disabled_cache.cache_dir, f"{file_hash}.json")
        assert not os.path.exists(cache_file_path)

        # Attempt to load from cache
        loaded_docs = disabled_cache.load_from_cache(file_hash)
        assert loaded_docs is None

    def test_load_nonexistent_cache_entry(self, cache_manager):
        """Test loading a cache entry that doesn't exist."""
        nonexistent_hash = "nonexistent_hash"

        # Attempt to load nonexistent entry
        result = cache_manager.load_from_cache(nonexistent_hash)

        # Verify None is returned
        assert result is None


class TestUtils:
    """Test suite for utility functions in the Gemini PDF Reader."""

    def test_is_web_url_with_valid_http_urls(self):
        """Test is_web_url with valid HTTP URLs."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://example.com/path/to/file.pdf",
            "http://example.com/path?query=value",
            "https://sub.domain.example.com:8080/path",
        ]

        for url in valid_urls:
            assert (
                is_web_url(url) is True
            ), f"Expected {url} to be recognized as a valid URL"

    def test_is_web_url_with_invalid_urls(self):
        """Test is_web_url with invalid URLs."""
        invalid_urls = [
            "",  # Empty string
            "file:///path/to/file.pdf",  # File scheme
            "ftp://example.com/file.pdf",  # FTP scheme
            "/path/to/file.pdf",  # Local path
            "C:\\path\\to\\file.pdf",  # Windows path
            "path/to/file.pdf",  # Relative path
            "://malformed.url",  # Malformed URL
        ]

        for url in invalid_urls:
            assert (
                is_web_url(url) is False
            ), f"Expected {url} to be recognized as an invalid URL"

    def test_is_web_url_with_exception(self):
        """Test is_web_url handles exceptions gracefully."""
        # Create a scenario where urlparse would raise an exception
        with patch("llama_index.readers.gemini.utils.urlparse") as mock_urlparse:
            mock_urlparse.side_effect = Exception("Test exception")
            assert is_web_url("http://example.com") is False

    @patch("llama_index.readers.gemini.utils.requests.get")
    def test_download_from_url_success(self, mock_get):
        """Test successful file download."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.iter_content.return_value = [b"mock", b"pdf", b"content"]
        mock_get.return_value = mock_response

        test_url = "https://arxiv.org/pdf/1706.03762"

        # Mock open function
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            # Call the function
            result = download_from_url(test_url, verbose=True)

            # Verify temp file path is returned
            assert os.path.basename(result).startswith("gemini_pdf_download_")
            assert result.endswith(".pdf")

            # Verify requests.get was called correctly
            mock_get.assert_called_once_with(test_url, stream=True, timeout=30)

            # Verify content was written to file
            assert mock_file().write.call_count == 3

    @patch("llama_index.readers.gemini.utils.requests.get")
    def test_download_from_url_with_non_pdf_content_type(self, mock_get):
        """Test download when Content-Type is not PDF
        but URL ends with .pdf.
        """
        # Setup mock response with non-PDF content type
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        mock_response.iter_content.return_value = [b"mock", b"content"]
        mock_get.return_value = mock_response

        test_url = "https://example.com/file.pdf"  # URL ends with .pdf

        # Mock open function
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            # Call the function
            result = download_from_url(test_url, verbose=True)

            # Verify temp file path is returned
            assert result.endswith(".pdf")

            # Verify content was written to file
            assert mock_file().write.call_count == 2

    @patch("llama_index.readers.gemini.utils.requests.get")
    @patch("llama_index.readers.gemini.utils.logger.warning")
    def test_download_warning_non_pdf(self, mock_warning, mock_get):
        """Test warning is logged when content might not be PDF."""
        # Setup mock response with non-PDF content type and
        # URL not ending with .pdf
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.iter_content.return_value = [b"mock", b"content"]
        mock_get.return_value = mock_response

        test_url = "https://example.com/file.html"  # URL doesn't end with .pdf

        # Mock open function
        with patch("builtins.open", new_callable=mock_open):
            # Call the function
            download_from_url(test_url, verbose=True)

            # Verify warning was logged
            mock_warning.assert_called_once()
            assert "may not be a PDF" in mock_warning.call_args[0][0]

    @patch("llama_index.readers.gemini.utils.requests.get")
    def test_download_from_url_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        # Setup mock response to raise HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        test_url = "https://example.com/file.pdf"

        # Call the function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            download_from_url(test_url)

        # Verify error message contains the original exception
        assert "Failed to download file" in str(exc_info.value)
        assert "HTTP Error" in str(exc_info.value)

    @patch("llama_index.readers.gemini.utils.requests.get")
    @patch("llama_index.readers.gemini.utils.logger.error")
    def test_download_from_url_other_error(self, mock_error, mock_get):
        """Test handling of other errors during download."""
        # Setup mock to raise connection error
        mock_get.side_effect = Exception("Connection failed")

        test_url = "https://example.com/file.pdf"

        # Call the function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            download_from_url(test_url, verbose=True)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error downloading from URL" in mock_error.call_args[0][0]

        # Verify error message contains the original exception
        assert "Failed to download file" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)

    def test_download_from_url_integration(self):
        """Integration test for download_from_url with a real URL."""
        # A known stable PDF URL for testing
        test_url = (
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        )

        try:
            # Attempt to download the file
            result_path = download_from_url(test_url)

            # Verify file was downloaded
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0

            # Clean up
            os.remove(result_path)
        except RuntimeError as e:
            pytest.fail(f"Download from URL failed: {e}")
