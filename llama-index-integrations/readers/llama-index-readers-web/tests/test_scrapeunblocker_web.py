"""Tests for ScrapeUnblockerWebReader."""

import json
from unittest.mock import Mock, patch

import pytest
import requests
from llama_index.core.schema import Document
from llama_index.readers.web import ScrapeUnblockerWebReader


@pytest.fixture
def api_key() -> str:
    """Test API key fixture."""
    return "test_api_key_123"


@pytest.fixture
def test_url() -> str:
    """Test URL fixture."""
    return "https://example.com"


@pytest.fixture
def mock_html_response() -> str:
    """Mock HTML response content."""
    return """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Welcome to Test Page</h1>
        <p>This is a test paragraph.</p>
    </body>
    </html>
    """


def _mock_response(text: str, status_code: int = 200, json_data=None) -> Mock:
    response = Mock(spec=requests.Response)
    response.text = text
    response.content = text.encode()
    response.status_code = status_code
    response.headers = {"Content-Type": "text/html"}
    response.raise_for_status = Mock()
    if json_data is not None:
        response.json = Mock(return_value=json_data)
    return response


def test_class_name(api_key: str) -> None:
    assert ScrapeUnblockerWebReader(api_key=api_key).class_name() == (
        "ScrapeUnblockerWebReader"
    )


def test_is_remote(api_key: str) -> None:
    assert ScrapeUnblockerWebReader(api_key=api_key).is_remote is True


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError):
        ScrapeUnblockerWebReader(api_key="")


def test_invalid_proxy_country_raises(api_key: str) -> None:
    with pytest.raises(ValueError):
        ScrapeUnblockerWebReader(api_key=api_key, proxy_country="usa")


def test_proxy_country_normalised(api_key: str) -> None:
    reader = ScrapeUnblockerWebReader(api_key=api_key, proxy_country="US")
    assert reader.proxy_country == "us"


def test_prepare_request_params_drops_unset(api_key: str, test_url: str) -> None:
    reader = ScrapeUnblockerWebReader(api_key=api_key)
    params = reader._prepare_request_params(test_url)
    assert params == {"url": test_url}


def test_prepare_request_params_includes_options(api_key: str, test_url: str) -> None:
    reader = ScrapeUnblockerWebReader(
        api_key=api_key, proxy_country="de", time_sleep=5, parsed_data=True
    )
    params = reader._prepare_request_params(test_url)
    assert params["url"] == test_url
    assert params["proxy_country"] == "de"
    assert params["time_sleep"] == 5
    assert params["parsed_data"] is True


def test_extra_params_override(api_key: str, test_url: str) -> None:
    reader = ScrapeUnblockerWebReader(api_key=api_key, proxy_country="de")
    params = reader._prepare_request_params(test_url, {"proxy_country": "fr"})
    assert params["proxy_country"] == "fr"


@patch("llama_index.readers.web.scrapeunblocker_web.base.requests.post")
def test_load_data_single_url(
    mock_post: Mock, api_key: str, test_url: str, mock_html_response: str
) -> None:
    mock_post.return_value = _mock_response(mock_html_response)
    reader = ScrapeUnblockerWebReader(api_key=api_key)

    documents = reader.load_data(test_url)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert "Welcome to Test Page" in documents[0].text
    assert documents[0].metadata["source_url"] == test_url
    assert documents[0].metadata["status_code"] == 200

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-ScrapeUnblocker-Key"] == api_key
    assert kwargs["params"]["url"] == test_url


@patch("llama_index.readers.web.scrapeunblocker_web.base.requests.post")
def test_load_data_multiple_urls(
    mock_post: Mock, api_key: str, mock_html_response: str
) -> None:
    mock_post.return_value = _mock_response(mock_html_response)
    reader = ScrapeUnblockerWebReader(api_key=api_key)

    documents = reader.load_data(["https://a.com", "https://b.com"])

    assert len(documents) == 2
    assert mock_post.call_count == 2


@patch("llama_index.readers.web.scrapeunblocker_web.base.requests.post")
def test_load_data_parsed_data_returns_json(
    mock_post: Mock, api_key: str, test_url: str
) -> None:
    payload = {"title": "Test Page", "price": "9.99"}
    mock_post.return_value = _mock_response("{}", json_data=payload)
    reader = ScrapeUnblockerWebReader(api_key=api_key, parsed_data=True)

    documents = reader.load_data(test_url)

    assert json.loads(documents[0].text) == payload


@patch("llama_index.readers.web.scrapeunblocker_web.base.requests.post")
def test_load_data_error_does_not_abort_batch(
    mock_post: Mock, api_key: str, mock_html_response: str
) -> None:
    ok = _mock_response(mock_html_response)
    failing = Mock(spec=requests.Response)
    failing.raise_for_status = Mock(side_effect=requests.HTTPError("403 Forbidden"))
    mock_post.side_effect = [failing, ok]

    reader = ScrapeUnblockerWebReader(api_key=api_key)
    documents = reader.load_data(["https://blocked.com", "https://ok.com"])

    assert len(documents) == 2
    assert documents[0].metadata["status"] == "failed"
    assert "403 Forbidden" in documents[0].metadata["error"]
    assert "Welcome to Test Page" in documents[1].text
