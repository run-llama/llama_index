"""Test SVG processing with optional dependencies."""

from unittest.mock import MagicMock, patch
import pytest


class TestSVGOptionalDependencies:
    """Test that SVG processing is optional and gracefully handles missing dependencies."""

    @patch("atlassian.Confluence")
    def test_svg_processing_without_svglib(self, mock_confluence_class):
        """Test that SVG processing returns empty string when svglib is not installed."""
        from llama_index.readers.confluence import ConfluenceReader

        # Mock the confluence client instance
        mock_confluence_instance = MagicMock()
        mock_confluence_class.return_value = mock_confluence_instance

        # Create reader
        reader = ConfluenceReader(
            base_url="https://test.atlassian.com/wiki", api_token="test_token"
        )

        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<svg>test</svg>"
        mock_confluence_instance.request.return_value = mock_response

        # Hide svglib and reportlab imports to simulate missing dependencies
        with patch.dict(
            "sys.modules",
            {
                "svglib": None,
                "svglib.svglib": None,
                "reportlab": None,
                "reportlab.graphics": None,
            },
        ):
            # Should return empty string and log warning instead of raising error
            result = reader.process_svg("test_link")
            assert result == ""

    @patch("atlassian.Confluence")
    def test_svg_processing_with_svglib_available(self, mock_confluence_class):
        """Test that SVG processing works when svglib is available."""
        # Skip this test if svglib is not actually installed
        try:
            import svglib  # noqa: F401
            from reportlab.graphics import renderPM  # noqa: F401
        except ImportError:
            pytest.skip("SVG dependencies not installed")

        from llama_index.readers.confluence import ConfluenceReader

        # Mock the confluence client instance
        mock_confluence_instance = MagicMock()
        mock_confluence_class.return_value = mock_confluence_instance

        reader = ConfluenceReader(
            base_url="https://test.atlassian.com/wiki", api_token="test_token"
        )

        # Create a minimal valid SVG
        svg_content = b"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <rect width="100" height="100" fill="white"/>
  <text x="10" y="50" fill="black">Test</text>
</svg>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = svg_content
        mock_confluence_instance.request.return_value = mock_response

        # Should process without error (actual text extraction may vary)
        result = reader.process_svg("test_link")
        # Result should be a string (may be empty if tesseract can't extract text)
        assert isinstance(result, str)

    @patch("atlassian.Confluence")
    def test_svg_processing_with_empty_response(self, mock_confluence_class):
        """Test that SVG processing handles empty responses gracefully."""
        from llama_index.readers.confluence import ConfluenceReader

        # Mock the confluence client instance
        mock_confluence_instance = MagicMock()
        mock_confluence_class.return_value = mock_confluence_instance

        reader = ConfluenceReader(
            base_url="https://test.atlassian.com/wiki", api_token="test_token"
        )

        # Test with empty content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b""
        mock_confluence_instance.request.return_value = mock_response

        result = reader.process_svg("test_link")
        assert result == ""

        # Test with None content
        mock_response.content = None
        result = reader.process_svg("test_link")
        assert result == ""

        # Test with non-200 status
        mock_response.status_code = 404
        mock_response.content = b"<svg>test</svg>"
        result = reader.process_svg("test_link")
        assert result == ""

    @patch("atlassian.Confluence")
    def test_reader_initialization_without_svglib(self, mock_confluence_class):
        """Test that ConfluenceReader can be initialized without svglib installed."""
        from llama_index.readers.confluence import ConfluenceReader

        # Mock the confluence client instance
        mock_confluence_instance = MagicMock()
        mock_confluence_class.return_value = mock_confluence_instance

        # Should not raise an error during initialization
        reader = ConfluenceReader(
            base_url="https://test.atlassian.com/wiki", api_token="test_token"
        )
        assert reader is not None
        assert reader.base_url == "https://test.atlassian.com/wiki"
