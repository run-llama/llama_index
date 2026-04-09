import unittest
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mrscraper import MrScraperToolSpec


class TestMrScraperToolSpec(unittest.TestCase):
    def test_class_inheritance(self):
        """Test that MrScraperToolSpec inherits from BaseToolSpec."""
        names_of_base_classes = [b.__name__ for b in MrScraperToolSpec.__mro__]
        self.assertIn(BaseToolSpec.__name__, names_of_base_classes)

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    def test_initialization(self, mock_mrscraper_cls):
        """Test that the class initializes correctly."""
        tool = MrScraperToolSpec(api_key="test_token")
        mock_mrscraper_cls.assert_called_once_with(token="test_token")
        self.assertFalse(tool._verbose)

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    def test_initialization_verbose(self, mock_mrscraper_cls):
        """Test verbose initialization."""
        tool = MrScraperToolSpec(api_key="test_token", verbose=True)
        self.assertTrue(tool._verbose)

    def test_spec_functions(self):
        """Test that spec_functions lists the expected tools."""
        expected = [
            "fetch_html",
            "create_scraper",
            "rerun_scraper",
            "bulk_rerun_ai_scraper",
            "rerun_manual_scraper",
            "bulk_rerun_manual_scraper",
            "get_all_results",
            "get_result_by_id",
        ]
        self.assertEqual(MrScraperToolSpec.spec_functions, expected)


@pytest.mark.asyncio
class TestMrScraperToolSpecAsync:
    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_fetch_html(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.fetch_html = AsyncMock(
            return_value={
                "status_code": 200,
                "data": "<html><body>Hello</body></html>",
                "headers": {},
            }
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        result = await tool.fetch_html("https://example.com")

        self.assertEqual = result.text, "<html><body>Hello</body></html>"
        assert result.metadata["url"] == "https://example.com"
        assert result.metadata["status_code"] == 200
        mock_client.fetch_html.assert_awaited_once_with(
            "https://example.com",
            timeout=120,
            geo_code="US",
            block_resources=False,
        )

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_fetch_html_with_options(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.fetch_html = AsyncMock(
            return_value={
                "status_code": 200,
                "data": "<html></html>",
                "headers": {},
            }
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        await tool.fetch_html(
            "https://example.com",
            timeout=60,
            geo_code="GB",
            block_resources=True,
        )

        mock_client.fetch_html.assert_awaited_once_with(
            "https://example.com",
            timeout=60,
            geo_code="GB",
            block_resources=True,
        )

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_create_scraper(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.create_scraper = AsyncMock(
            return_value={
                "status_code": 200,
                "data": {"id": "scraper_123"},
                "headers": {},
            }
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        result = await tool.create_scraper(
            "https://example.com/products",
            "Extract all product names and prices",
            agent="listing",
            proxy_country="US",
        )

        assert result["data"]["id"] == "scraper_123"
        mock_client.create_scraper.assert_awaited_once()

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_rerun_scraper(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.rerun_scraper = AsyncMock(
            return_value={"status_code": 200, "data": {}, "headers": {}}
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        await tool.rerun_scraper("scraper_123", "https://example.com/page2")

        mock_client.rerun_scraper.assert_awaited_once()

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_bulk_rerun_ai_scraper(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.bulk_rerun_ai_scraper = AsyncMock(
            return_value={"status_code": 200, "data": {}, "headers": {}}
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        urls = ["https://example.com/1", "https://example.com/2"]
        await tool.bulk_rerun_ai_scraper("scraper_123", urls)

        mock_client.bulk_rerun_ai_scraper.assert_awaited_once_with("scraper_123", urls)

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_rerun_manual_scraper(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.rerun_manual_scraper = AsyncMock(
            return_value={"status_code": 200, "data": {}, "headers": {}}
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        await tool.rerun_manual_scraper("manual_123", "https://example.com/item")

        mock_client.rerun_manual_scraper.assert_awaited_once_with(
            "manual_123", "https://example.com/item"
        )

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_bulk_rerun_manual_scraper(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.bulk_rerun_manual_scraper = AsyncMock(
            return_value={"status_code": 200, "data": {}, "headers": {}}
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        urls = ["https://example.com/a", "https://example.com/b"]
        await tool.bulk_rerun_manual_scraper("manual_123", urls)

        mock_client.bulk_rerun_manual_scraper.assert_awaited_once_with(
            "manual_123", urls
        )

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_get_all_results(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.get_all_results = AsyncMock(
            return_value={
                "status_code": 200,
                "data": {"results": [], "total": 0},
                "headers": {},
            }
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        result = await tool.get_all_results(page_size=5, page=2)

        assert result["data"]["total"] == 0
        mock_client.get_all_results.assert_awaited_once()

    @patch("llama_index.tools.mrscraper.base.MrScraper")
    async def test_get_result_by_id(self, mock_mrscraper_cls):
        mock_client = MagicMock()
        mock_client.get_result_by_id = AsyncMock(
            return_value={
                "status_code": 200,
                "data": {"id": "result_456", "status": "completed"},
                "headers": {},
            }
        )
        mock_mrscraper_cls.return_value = mock_client

        tool = MrScraperToolSpec(api_key="test_token")
        result = await tool.get_result_by_id("result_456")

        assert result["data"]["id"] == "result_456"
        mock_client.get_result_by_id.assert_awaited_once_with("result_456")


if __name__ == "__main__":
    unittest.main()
