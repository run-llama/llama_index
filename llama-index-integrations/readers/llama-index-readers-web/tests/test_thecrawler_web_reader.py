"""Tests for TheCrawlerWebReader."""

from unittest.mock import Mock, patch

import pytest

from llama_index.readers.web import TheCrawlerWebReader


@pytest.fixture
def api_key() -> str:
    return "mai_live_" + "a" * 32


def _success_payload() -> dict:
    return {
        "pages": [
            {
                "url": "https://example.com",
                "title": "Example Domain",
                "description": "Illustrative example",
                "language": "en",
                "canonicalUrl": "https://example.com/",
                "markdown": "# Example Domain\n\nIllustrative example.",
                "text": "Example Domain\n\nIllustrative example.",
                "statusCode": 200,
                "contentType": "text/html",
                "responseTimeMs": 123,
                "scrapedAt": "2026-01-01T00:00:00Z",
                "status": "success",
                "error": None,
                "errorType": None,
                "errorRetryable": False,
                "fromCache": False,
            }
        ],
        "totalScraped": 1,
        "totalErrors": 0,
        "durationMs": 123,
    }


def _error_payload() -> dict:
    return {
        "pages": [
            {
                "url": "https://does-not-exist.invalid",
                "title": None,
                "description": None,
                "markdown": None,
                "text": None,
                "statusCode": 0,
                "status": "error",
                "error": "ENOTFOUND does-not-exist.invalid",
                "errorType": "dns",
                "errorRetryable": False,
                "fromCache": False,
                "scrapedAt": "2026-01-01T00:00:00Z",
            }
        ],
        "totalScraped": 0,
        "totalErrors": 1,
        "durationMs": 50,
    }


class TestTheCrawlerWebReader:
    def test_class_name(self, api_key):
        assert (
            TheCrawlerWebReader(api_key=api_key).class_name() == "TheCrawlerWebReader"
        )

    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("THECRAWLER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TheCrawler API key is required"):
            TheCrawlerWebReader(api_key=None)

    def test_init_picks_up_env(self, monkeypatch):
        monkeypatch.setenv("THECRAWLER_API_KEY", "mai_live_envvalue")
        reader = TheCrawlerWebReader()
        assert reader.api_key == "mai_live_envvalue"

    def test_api_url_defaults_and_strips_trailing_slash(self, api_key):
        reader = TheCrawlerWebReader(api_key=api_key)
        assert reader.api_url == "https://www.miaibot.ai/api/v1"
        reader2 = TheCrawlerWebReader(
            api_key=api_key, api_url="http://localhost:3000/v1/"
        )
        assert reader2.api_url == "http://localhost:3000/v1"

    @patch("llama_index.readers.web.thecrawler_web.base.requests.post")
    def test_load_data_success_builds_document(self, mock_post, api_key):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _success_payload()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        reader = TheCrawlerWebReader(api_key=api_key)
        docs = reader.load_data(urls=["https://example.com"])

        assert len(docs) == 1
        doc = docs[0]
        assert doc.text.startswith("# Example Domain")
        assert doc.metadata["source"] == "thecrawler"
        assert doc.metadata["url"] == "https://example.com"
        assert doc.metadata["title"] == "Example Domain"
        assert doc.metadata["status"] == "success"
        assert doc.metadata["status_code"] == 200
        # error_* keys must not be present on success
        assert "error_type" not in doc.metadata
        assert "error_retryable" not in doc.metadata

        # Verify request was constructed correctly
        call_args = mock_post.call_args
        assert call_args.args[0] == "https://www.miaibot.ai/api/v1/crawl"
        assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {api_key}"
        body = call_args.kwargs["json"]
        assert body["urls"] == ["https://example.com"]
        assert body["extractMarkdown"] is True
        assert body["stripBoilerplate"] is True

    @patch("llama_index.readers.web.thecrawler_web.base.requests.post")
    def test_load_data_error_page_does_not_throw(self, mock_post, api_key):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _error_payload()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        reader = TheCrawlerWebReader(api_key=api_key)
        docs = reader.load_data(urls=["https://does-not-exist.invalid"])

        assert len(docs) == 1
        doc = docs[0]
        assert doc.text == ""
        assert doc.metadata["status"] == "error"
        assert doc.metadata["error_type"] == "dns"
        assert doc.metadata["error_retryable"] is False
        assert "error" in doc.metadata

    @patch("llama_index.readers.web.thecrawler_web.base.requests.post")
    def test_load_data_forwards_extra_params(self, mock_post, api_key):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"pages": []}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        reader = TheCrawlerWebReader(
            api_key=api_key, params={"usePlaywright": True, "requestTimeoutSecs": 60}
        )
        reader.load_data(urls=["https://example.com"])

        body = mock_post.call_args.kwargs["json"]
        assert body["usePlaywright"] is True
        assert body["requestTimeoutSecs"] == 60
        # defaults must still be present
        assert body["extractMarkdown"] is True
        assert body["stripBoilerplate"] is True

    def test_load_data_empty_urls_raises(self, api_key):
        reader = TheCrawlerWebReader(api_key=api_key)
        with pytest.raises(ValueError):
            reader.load_data(urls=[])

    @patch("llama_index.readers.web.thecrawler_web.base.requests.post")
    def test_load_data_preserves_input_order(self, mock_post, api_key):
        payload = {
            "pages": [
                {"url": "https://a", "markdown": "A", "status": "success"},
                {"url": "https://b", "markdown": "B", "status": "success"},
                {"url": "https://c", "markdown": "C", "status": "success"},
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        reader = TheCrawlerWebReader(api_key=api_key)
        docs = reader.load_data(urls=["https://a", "https://b", "https://c"])

        assert [d.text for d in docs] == ["A", "B", "C"]
        assert [d.metadata["url"] for d in docs] == [
            "https://a",
            "https://b",
            "https://c",
        ]
