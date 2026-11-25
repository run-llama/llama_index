import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from llama_index.core.readers.base import BaseReader
from llama_index.readers.zendesk import ZendeskReader


def test_class():
    names_of_base_classes = [b.__name__ for b in ZendeskReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_success():
    article1 = {
        "id": 1,
        "title": "Article One",
        "html_url": "https://example.zendesk.com/hc/en-us/articles/1",
        "updated_at": "2026-01-01T00:00:00Z",
        "body": "<p>Hello <b>World</b></p>",
    }
    article2 = {
        "id": 2,
        "title": "Article Two",
        "html_url": "https://example.zendesk.com/hc/en-us/articles/2",
        "updated_at": "2026-01-02T00:00:00Z",
        "body": "Plain text body",
    }
    resp1 = MagicMock()
    resp1.text = json.dumps(
        {
            "articles": [article1],
            "next_page": "https://example.zendesk.com/api/v2/help_center/en-us/articles?page=2",
        }
    )
    resp2 = MagicMock()
    resp2.text = json.dumps({"articles": [article2], "next_page": None})

    with patch("requests.get", side_effect=[resp1, resp2]) as mock_get:
        reader = ZendeskReader(zendesk_subdomain="example")
        docs = reader.load_data()

    assert len(docs) == 2
    assert docs[0].text == "Hello World"
    assert docs[0].metadata == {
        "id": 1,
        "title": "Article One",
        "url": "https://example.zendesk.com/hc/en-us/articles/1",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    assert docs[1].text == "Plain text body"
    assert docs[1].metadata["id"] == 2
    assert resp1.raise_for_status.call_count == 1
    assert resp2.raise_for_status.call_count == 1
    assert mock_get.call_count == 2


def test_load_data_raises_on_http_error():
    article = {
        "id": 100,
        "title": "First",
        "html_url": "https://example.zendesk.com/100",
        "updated_at": "2026-01-01T00:00:00Z",
        "body": "<p>Text</p>",
    }
    resp1 = MagicMock()
    resp1.text = json.dumps(
        {
            "articles": [article],
            "next_page": "https://example.zendesk.com/page2",
        }
    )

    resp2 = MagicMock()
    resp2.status_code = 429
    resp2.text = json.dumps({"error": "RateLimited"})
    resp2.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "429 Client Error: Rate Limited"
    )

    with patch("requests.get", side_effect=[resp1, resp2]):
        reader = ZendeskReader(zendesk_subdomain="example")
        with pytest.raises(requests.exceptions.HTTPError, match="429 Client Error"):
            reader.load_data()


def test_get_articles_page_timeout():
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"articles": [], "next_page": None})

    with patch("requests.get", return_value=mock_resp) as mock_get:
        reader = ZendeskReader(zendesk_subdomain="example", timeout=15.0)
        reader.get_articles_page()

    mock_get.assert_called_once_with(
        "https://example.zendesk.com/api/v2/help_center/en-us/articles?per_page=100",
        timeout=15.0,
    )


def test_get_articles_page_invalid_json():
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(["not", "a", "dict"])

    with patch("requests.get", return_value=mock_resp):
        reader = ZendeskReader(zendesk_subdomain="example")
        with pytest.raises(ValueError, match="Expected JSON object"):
            reader.get_articles_page()


def test_load_data_skips_none_body():
    article = {
        "id": 10,
        "title": "Empty",
        "html_url": "https://example.zendesk.com/10",
        "updated_at": "2026-01-01T00:00:00Z",
        "body": None,
    }
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"articles": [article], "next_page": None})

    with patch("requests.get", return_value=mock_resp):
        reader = ZendeskReader(zendesk_subdomain="example")
        docs = reader.load_data()

    assert len(docs) == 0
