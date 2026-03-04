# tests/test_tools_brightdata.py
import unittest
from unittest.mock import patch, MagicMock
import json
import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.brightdata import BrightDataToolSpec


class TestBrightDataToolSpec(unittest.TestCase):
    def test_class_inheritance(self):
        """Test that BrightDataToolSpec inherits from BaseToolSpec."""
        names_of_base_classes = [b.__name__ for b in BrightDataToolSpec.__mro__]
        self.assertIn(BaseToolSpec.__name__, names_of_base_classes)

    def test_initialization(self):
        """Test that the class initializes correctly."""
        tool = BrightDataToolSpec(api_key="test_key", zone="test_zone")
        self.assertEqual(tool._api_key, "test_key")
        self.assertEqual(tool._zone, "test_zone")
        self.assertEqual(tool._endpoint, "https://api.brightdata.com/request")

    @patch("requests.post")
    def test_scrape_as_markdown_success(self, mock_post):
        """Test successful scraping."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Markdown Content\n\nThis is a test."
        mock_post.return_value = mock_response

        tool = BrightDataToolSpec(api_key="test_key")
        result = tool.scrape_as_markdown("https://example.com")

        self.assertEqual(result.text, "# Markdown Content\n\nThis is a test.")
        self.assertEqual(result.metadata, {"url": "https://example.com"})

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://api.brightdata.com/request")

        payload = json.loads(call_args[1]["data"])
        self.assertEqual(payload["url"], "https://example.com")
        self.assertEqual(payload["zone"], "unblocker")  # default value
        self.assertEqual(payload["format"], "raw")
        self.assertEqual(payload["data_format"], "markdown")

        headers = call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test_key")

    @patch("requests.post")
    def test_scrape_as_markdown_failure(self, mock_post):
        """Test failed scraping."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Access denied"
        mock_post.return_value = mock_response

        tool = BrightDataToolSpec(api_key="test_key")

        with pytest.raises(Exception) as context:
            tool.scrape_as_markdown("https://example.com")

        self.assertIn("Failed to scrape: 403", str(context.value))


if __name__ == "__main__":
    unittest.main()
