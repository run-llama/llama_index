"""Tests for MrScraperWebReader."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.schema import Document
from llama_index.readers.web.mrscraper_web.base import MrScraperWebReader

MOCK_TOKEN = ""


@pytest.fixture()
def _patch_sdk():
    """Patch the MrScraper SDK import so tests don't need the real package."""
    mock_client = AsyncMock()
    mock_cls = MagicMock(return_value=mock_client)
    with patch.dict("sys.modules", {"mrscraper": MagicMock(MrScraper=mock_cls)}):
        yield mock_client


@pytest.fixture()
def mock_client(_patch_sdk):
    return _patch_sdk


def _make_reader(mode: str = "fetch_html", **kwargs):
    """Create a reader with patched SDK."""
    return MrScraperWebReader(api_token=MOCK_TOKEN, mode=mode, **kwargs)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_with_valid_token(self, _patch_sdk):
        reader = _make_reader()
        assert reader.api_token == MOCK_TOKEN
        assert reader.mode == "fetch_html"

    def test_init_with_empty_token_raises(self, _patch_sdk):
        with pytest.raises(ValueError, match="API token is required"):
            MrScraperWebReader(api_token="", mode="fetch_html")

    def test_init_with_invalid_mode_raises(self, _patch_sdk):
        with pytest.raises(ValueError, match="Invalid mode"):
            MrScraperWebReader(api_token=MOCK_TOKEN, mode="invalid_mode")

    @pytest.mark.parametrize(
        "mode",
        [
            "fetch_html",
            "scrape",
            "rerun_scraper",
            "bulk_rerun_ai_scraper",
            "rerun_manual_scraper",
            "bulk_rerun_manual_scraper",
            "get_all_results",
            "get_result_by_id",
        ],
    )
    def test_init_all_valid_modes(self, _patch_sdk, mode):
        reader = MrScraperWebReader(api_token=MOCK_TOKEN, mode=mode)
        assert reader.mode == mode

    def test_class_name(self, _patch_sdk):
        reader = _make_reader()
        assert reader.class_name() == "MrScraperWebReader"

    def test_import_error_without_sdk(self):
        with patch.dict("sys.modules", {"mrscraper": None}):
            with pytest.raises(ImportError, match="mrscraper-sdk not found"):
                MrScraperWebReader(api_token=MOCK_TOKEN)


# ---------------------------------------------------------------------------
# fetch_html mode tests
# ---------------------------------------------------------------------------


class TestFetchHtml:
    @pytest.mark.asyncio
    async def test_aload_data_fetch_html(self, mock_client):
        mock_client.fetch_html.return_value = {
            "status_code": 200,
            "data": "<html><body>Hello</body></html>",
            "headers": {},
        }
        reader = _make_reader(mode="fetch_html")
        docs = await reader.aload_data(url="https://example.com")

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Hello" in docs[0].text
        assert docs[0].metadata["source_url"] == "https://example.com"
        assert docs[0].metadata["status_code"] == 200
        assert docs[0].metadata["mode"] == "fetch_html"
        mock_client.fetch_html.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_html_with_params(self, mock_client):
        mock_client.fetch_html.return_value = {
            "status_code": 200,
            "data": "<html></html>",
            "headers": {},
        }
        reader = _make_reader(mode="fetch_html")
        docs = await reader.aload_data(
            url="https://example.com",
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
        assert docs[0].metadata["geo_code"] == "GB"

    @pytest.mark.asyncio
    async def test_fetch_html_missing_url_raises(self, mock_client):
        reader = _make_reader(mode="fetch_html")
        with pytest.raises(ValueError, match="url is required"):
            await reader.aload_data()

    @pytest.mark.asyncio
    async def test_standalone_fetch_html(self, mock_client):
        mock_client.fetch_html.return_value = {
            "status_code": 200,
            "data": "<html>standalone</html>",
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.fetch_html("https://example.com", geo_code="US")
        assert len(docs) == 1
        assert "standalone" in docs[0].text


# ---------------------------------------------------------------------------
# scrape (create_scraper) mode tests
# ---------------------------------------------------------------------------


class TestCreateScraper:
    @pytest.mark.asyncio
    async def test_aload_data_scrape(self, mock_client):
        mock_client.create_scraper.return_value = {
            "status_code": 200,
            "data": {"id": "scraper_99", "result": [{"name": "Widget", "price": 9.99}]},
            "headers": {},
        }
        reader = _make_reader(mode="scrape")
        docs = await reader.aload_data(
            url="https://example.com/products",
            message="Extract product names and prices",
            agent="listing",
            proxy_country="US",
        )

        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "scrape"
        assert docs[0].metadata["scraper_id"] == "scraper_99"
        assert docs[0].metadata["agent"] == "listing"
        assert docs[0].metadata["message"] == "Extract product names and prices"
        mock_client.create_scraper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_scrape_missing_url_raises(self, mock_client):
        reader = _make_reader(mode="scrape")
        with pytest.raises(ValueError, match="url is required"):
            await reader.aload_data(message="Extract stuff")

    @pytest.mark.asyncio
    async def test_scrape_missing_message_raises(self, mock_client):
        reader = _make_reader(mode="scrape")
        with pytest.raises(ValueError, match="message is required"):
            await reader.aload_data(url="https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_map_agent(self, mock_client):
        mock_client.create_scraper.return_value = {
            "status_code": 200,
            "data": {"id": "scraper_map"},
            "headers": {},
        }
        reader = _make_reader(mode="scrape")
        docs = await reader.aload_data(
            url="https://example.com",
            message="Crawl all pages",
            agent="map",
            max_depth=3,
            max_pages=100,
            limit=500,
            include_patterns="/products/.*",
            exclude_patterns="/admin/.*",
        )
        call_kwargs = mock_client.create_scraper.call_args
        assert call_kwargs.kwargs["agent"] == "map"
        assert call_kwargs.kwargs["max_depth"] == 3
        assert call_kwargs.kwargs["max_pages"] == 100

    @pytest.mark.asyncio
    async def test_standalone_create_scraper(self, mock_client):
        mock_client.create_scraper.return_value = {
            "status_code": 200,
            "data": {"id": "sc_1"},
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.create_scraper("https://example.com", "Extract titles")
        assert len(docs) == 1
        assert docs[0].metadata["scraper_id"] == "sc_1"


# ---------------------------------------------------------------------------
# rerun_scraper mode tests
# ---------------------------------------------------------------------------


class TestRerunScraper:
    @pytest.mark.asyncio
    async def test_aload_data_rerun_scraper(self, mock_client):
        mock_client.rerun_scraper.return_value = {
            "status_code": 200,
            "data": {"result": "rerun data"},
            "headers": {},
        }
        reader = _make_reader(mode="rerun_scraper")
        docs = await reader.aload_data(
            scraper_id="scraper_1",
            url="https://example.com/page2",
            max_depth=3,
        )

        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "rerun_scraper"
        assert docs[0].metadata["scraper_id"] == "scraper_1"
        mock_client.rerun_scraper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rerun_scraper_missing_scraper_id(self, mock_client):
        reader = _make_reader(mode="rerun_scraper")
        with pytest.raises(ValueError, match="scraper_id is required"):
            await reader.aload_data(url="https://example.com")

    @pytest.mark.asyncio
    async def test_rerun_scraper_missing_url(self, mock_client):
        reader = _make_reader(mode="rerun_scraper")
        with pytest.raises(ValueError, match="url is required"):
            await reader.aload_data(scraper_id="sc_1")

    @pytest.mark.asyncio
    async def test_standalone_rerun_scraper(self, mock_client):
        mock_client.rerun_scraper.return_value = {
            "status_code": 200,
            "data": "rerun html",
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.rerun_scraper("sc_1", "https://example.com")
        assert len(docs) == 1
        assert docs[0].metadata["scraper_id"] == "sc_1"


# ---------------------------------------------------------------------------
# bulk_rerun_ai_scraper mode tests
# ---------------------------------------------------------------------------


class TestBulkRerunAiScraper:
    @pytest.mark.asyncio
    async def test_aload_data_bulk_rerun_ai(self, mock_client):
        mock_client.bulk_rerun_ai_scraper.return_value = {
            "status_code": 200,
            "data": {"job_id": "bulk_1", "status": "queued"},
            "headers": {},
        }
        reader = _make_reader(mode="bulk_rerun_ai_scraper")
        urls = ["https://example.com/a", "https://example.com/b"]
        docs = await reader.aload_data(scraper_id="sc_1", urls=urls)

        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "bulk_rerun_ai_scraper"
        assert docs[0].metadata["source_urls"] == urls
        mock_client.bulk_rerun_ai_scraper.assert_awaited_once_with("sc_1", urls)

    @pytest.mark.asyncio
    async def test_bulk_ai_missing_scraper_id(self, mock_client):
        reader = _make_reader(mode="bulk_rerun_ai_scraper")
        with pytest.raises(ValueError, match="scraper_id is required"):
            await reader.aload_data(urls=["https://example.com"])

    @pytest.mark.asyncio
    async def test_bulk_ai_missing_urls(self, mock_client):
        reader = _make_reader(mode="bulk_rerun_ai_scraper")
        with pytest.raises(ValueError, match="urls.*required"):
            await reader.aload_data(scraper_id="sc_1")

    @pytest.mark.asyncio
    async def test_bulk_ai_empty_urls(self, mock_client):
        reader = _make_reader(mode="bulk_rerun_ai_scraper")
        with pytest.raises(ValueError, match="urls.*required"):
            await reader.aload_data(scraper_id="sc_1", urls=[])

    @pytest.mark.asyncio
    async def test_standalone_bulk_rerun_ai_scraper(self, mock_client):
        mock_client.bulk_rerun_ai_scraper.return_value = {
            "status_code": 200,
            "data": {"queued": True},
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.bulk_rerun_ai_scraper(
            "sc_1", ["https://example.com/1", "https://example.com/2"]
        )
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# rerun_manual_scraper mode tests
# ---------------------------------------------------------------------------


class TestRerunManualScraper:
    @pytest.mark.asyncio
    async def test_aload_data_rerun_manual(self, mock_client):
        mock_client.rerun_manual_scraper.return_value = {
            "status_code": 200,
            "data": {"result": "manual data"},
            "headers": {},
        }
        reader = _make_reader(mode="rerun_manual_scraper")
        docs = await reader.aload_data(
            scraper_id="manual_sc_1",
            url="https://example.com/product",
        )

        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "rerun_manual_scraper"
        assert docs[0].metadata["scraper_id"] == "manual_sc_1"
        mock_client.rerun_manual_scraper.assert_awaited_once_with(
            "manual_sc_1", "https://example.com/product"
        )

    @pytest.mark.asyncio
    async def test_rerun_manual_missing_scraper_id(self, mock_client):
        reader = _make_reader(mode="rerun_manual_scraper")
        with pytest.raises(ValueError, match="scraper_id is required"):
            await reader.aload_data(url="https://example.com")

    @pytest.mark.asyncio
    async def test_rerun_manual_missing_url(self, mock_client):
        reader = _make_reader(mode="rerun_manual_scraper")
        with pytest.raises(ValueError, match="url is required"):
            await reader.aload_data(scraper_id="manual_sc_1")

    @pytest.mark.asyncio
    async def test_standalone_rerun_manual_scraper(self, mock_client):
        mock_client.rerun_manual_scraper.return_value = {
            "status_code": 200,
            "data": "manual result",
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.rerun_manual_scraper("manual_sc_1", "https://example.com")
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# bulk_rerun_manual_scraper mode tests
# ---------------------------------------------------------------------------


class TestBulkRerunManualScraper:
    @pytest.mark.asyncio
    async def test_aload_data_bulk_manual(self, mock_client):
        mock_client.bulk_rerun_manual_scraper.return_value = {
            "status_code": 200,
            "data": {"job_id": "bulk_manual_1"},
            "headers": {},
        }
        reader = _make_reader(mode="bulk_rerun_manual_scraper")
        urls = ["https://example.com/x", "https://example.com/y"]
        docs = await reader.aload_data(scraper_id="msc_1", urls=urls)

        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "bulk_rerun_manual_scraper"
        assert docs[0].metadata["source_urls"] == urls
        mock_client.bulk_rerun_manual_scraper.assert_awaited_once_with("msc_1", urls)

    @pytest.mark.asyncio
    async def test_bulk_manual_missing_scraper_id(self, mock_client):
        reader = _make_reader(mode="bulk_rerun_manual_scraper")
        with pytest.raises(ValueError, match="scraper_id is required"):
            await reader.aload_data(urls=["https://example.com"])

    @pytest.mark.asyncio
    async def test_bulk_manual_missing_urls(self, mock_client):
        reader = _make_reader(mode="bulk_rerun_manual_scraper")
        with pytest.raises(ValueError, match="urls.*required"):
            await reader.aload_data(scraper_id="msc_1")

    @pytest.mark.asyncio
    async def test_standalone_bulk_rerun_manual_scraper(self, mock_client):
        mock_client.bulk_rerun_manual_scraper.return_value = {
            "status_code": 200,
            "data": {"ok": True},
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.bulk_rerun_manual_scraper(
            "msc_1", ["https://example.com/1"]
        )
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# get_all_results mode tests
# ---------------------------------------------------------------------------


class TestGetAllResults:
    @pytest.mark.asyncio
    async def test_aload_data_get_all_results_with_items(self, mock_client):
        mock_client.get_all_results.return_value = {
            "status_code": 200,
            "data": {
                "data": [
                    {"id": "r1", "url": "https://a.com", "status": "done"},
                    {"id": "r2", "url": "https://b.com", "status": "done"},
                ],
                "total": 2,
            },
            "headers": {},
        }
        reader = _make_reader(mode="get_all_results")
        docs = await reader.aload_data(page_size=20, sort_order="ASC")

        assert len(docs) == 2
        assert docs[0].metadata["result_id"] == "r1"
        assert docs[1].metadata["result_id"] == "r2"
        for doc in docs:
            assert doc.metadata["mode"] == "get_all_results"

    @pytest.mark.asyncio
    async def test_aload_data_get_all_results_empty(self, mock_client):
        mock_client.get_all_results.return_value = {
            "status_code": 200,
            "data": {"data": [], "total": 0},
            "headers": {},
        }
        reader = _make_reader(mode="get_all_results")
        docs = await reader.aload_data()
        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "get_all_results"

    @pytest.mark.asyncio
    async def test_get_all_results_with_filters(self, mock_client):
        mock_client.get_all_results.return_value = {
            "status_code": 200,
            "data": {"data": [], "total": 0},
            "headers": {},
        }
        reader = _make_reader(mode="get_all_results")
        await reader.aload_data(
            sort_field="createdAt",
            sort_order="ASC",
            page_size=5,
            page=2,
            search="product",
            date_range_column="createdAt",
            start_at="2024-01-01",
            end_at="2024-12-31",
        )
        mock_client.get_all_results.assert_awaited_once_with(
            sort_field="createdAt",
            sort_order="ASC",
            page_size=5,
            page=2,
            search="product",
            date_range_column="createdAt",
            start_at="2024-01-01",
            end_at="2024-12-31",
        )

    @pytest.mark.asyncio
    async def test_standalone_get_all_results(self, mock_client):
        mock_client.get_all_results.return_value = {
            "status_code": 200,
            "data": {
                "data": [{"id": "r1", "url": "https://x.com"}],
                "total": 1,
            },
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.get_all_results(page_size=5)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# get_result_by_id mode tests
# ---------------------------------------------------------------------------


class TestGetResultById:
    @pytest.mark.asyncio
    async def test_aload_data_get_result_by_id(self, mock_client):
        mock_client.get_result_by_id.return_value = {
            "status_code": 200,
            "data": {
                "id": "result_555",
                "url": "https://example.com",
                "content": "scraped content here",
            },
            "headers": {},
        }
        reader = _make_reader(mode="get_result_by_id")
        docs = await reader.aload_data(result_id="result_555")

        assert len(docs) == 1
        assert docs[0].metadata["mode"] == "get_result_by_id"
        assert docs[0].metadata["result_id"] == "result_555"
        assert "scraped content here" in docs[0].text
        mock_client.get_result_by_id.assert_awaited_once_with("result_555")

    @pytest.mark.asyncio
    async def test_get_result_by_id_missing_id(self, mock_client):
        reader = _make_reader(mode="get_result_by_id")
        with pytest.raises(ValueError, match="result_id is required"):
            await reader.aload_data()

    @pytest.mark.asyncio
    async def test_standalone_get_result_by_id(self, mock_client):
        mock_client.get_result_by_id.return_value = {
            "status_code": 200,
            "data": {"id": "r1", "content": "data"},
            "headers": {},
        }
        reader = _make_reader()
        docs = await reader.get_result_by_id("r1")
        assert len(docs) == 1
        assert docs[0].metadata["result_id"] == "r1"


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_text_string(self, _patch_sdk):
        result = {"data": "plain text content"}
        assert MrScraperWebReader._extract_text(result) == "plain text content"

    def test_extract_text_dict(self, _patch_sdk):
        data = {"key": "value", "nested": {"a": 1}}
        result = {"data": data}
        text = MrScraperWebReader._extract_text(result)
        parsed = json.loads(text)
        assert parsed["key"] == "value"

    def test_extract_text_list(self, _patch_sdk):
        data = [{"name": "A"}, {"name": "B"}]
        result = {"data": data}
        text = MrScraperWebReader._extract_text(result)
        parsed = json.loads(text)
        assert len(parsed) == 2

    def test_extract_text_missing_data(self, _patch_sdk):
        result = {}
        text = MrScraperWebReader._extract_text(result)
        assert text == ""

    def test_format_result_item_dict(self, _patch_sdk):
        item = {"id": "r1", "url": "https://x.com"}
        text = MrScraperWebReader._format_result_item(item)
        parsed = json.loads(text)
        assert parsed["id"] == "r1"

    def test_format_result_item_string(self, _patch_sdk):
        assert MrScraperWebReader._format_result_item("plain") == "plain"


# ---------------------------------------------------------------------------
# Data-type response tests
# ---------------------------------------------------------------------------


class TestResponseDataTypes:
    @pytest.mark.asyncio
    async def test_fetch_html_returns_html_string(self, mock_client):
        html = (
            "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
        )
        mock_client.fetch_html.return_value = {
            "status_code": 200,
            "data": html,
            "headers": {"content-type": "text/html"},
        }
        reader = _make_reader(mode="fetch_html")
        docs = await reader.aload_data(url="https://example.com")
        assert docs[0].text == html

    @pytest.mark.asyncio
    async def test_scrape_returns_json_data(self, mock_client):
        mock_client.create_scraper.return_value = {
            "status_code": 200,
            "data": {
                "id": "sc_1",
                "result": [
                    {"title": "Product A", "price": "$10"},
                    {"title": "Product B", "price": "$20"},
                ],
            },
            "headers": {},
        }
        reader = _make_reader(mode="scrape")
        docs = await reader.aload_data(
            url="https://example.com",
            message="Extract products",
        )
        parsed = json.loads(docs[0].text)
        assert "result" in parsed
        assert len(parsed["result"]) == 2

    @pytest.mark.asyncio
    async def test_fetch_html_non_200_status(self, mock_client):
        mock_client.fetch_html.return_value = {
            "status_code": 403,
            "data": "Forbidden",
            "headers": {},
        }
        reader = _make_reader(mode="fetch_html")
        docs = await reader.aload_data(url="https://example.com")
        assert docs[0].metadata["status_code"] == 403
        assert docs[0].text == "Forbidden"
