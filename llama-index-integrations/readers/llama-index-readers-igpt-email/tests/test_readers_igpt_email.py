"""Tests for IGPTEmailReader."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.igpt_email import IGPTEmailReader


def test_class():
    names_of_base_classes = [b.__name__ for b in IGPTEmailReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_returns_documents(mock_igpt_class):
    mock_results = [
        {
            "id": "msg-1",
            "subject": "Vendor proposal",
            "content": "Please find attached the vendor proposal for review.",
            "from": "vendor@external.com",
            "to": ["procurement@company.com"],
            "date": "2025-02-05",
            "thread_id": "thread-001",
        },
        {
            "id": "msg-2",
            "subject": "Re: Vendor proposal",
            "content": "We'll review and get back to you by Friday.",
            "from": "procurement@company.com",
            "to": ["vendor@external.com"],
            "date": "2025-02-06",
            "thread_id": "thread-001",
        },
    ]

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_results
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(
        query="vendor proposal",
        date_from="2025-02-01",
        date_to="2025-02-28",
        max_results=20,
    )

    mock_igpt_class.assert_called_once_with(api_key="test-key", user="test-user")
    mock_client.recall.search.assert_called_once_with(
        query="vendor proposal",
        date_from="2025-02-01",
        date_to="2025-02-28",
        max_results=20,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

    assert results[0].text == "Please find attached the vendor proposal for review."
    assert results[0].metadata["subject"] == "Vendor proposal"
    assert results[0].metadata["from"] == "vendor@external.com"
    assert results[0].metadata["to"] == ["procurement@company.com"]
    assert results[0].metadata["date"] == "2025-02-05"
    assert results[0].metadata["thread_id"] == "thread-001"
    assert results[0].metadata["id"] == "msg-1"
    assert results[0].metadata["source"] == "igpt_email"

    assert results[1].text == "We'll review and get back to you by Friday."
    assert results[1].metadata["id"] == "msg-2"


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_with_results_wrapper(mock_igpt_class):
    """Test load_data() handles a dict response with a 'results' key."""
    mock_response = {
        "results": [
            {
                "id": "msg-3",
                "subject": "Support ticket #42",
                "content": "Customer is unable to log in.",
                "from": "support@company.com",
                "to": ["tech@company.com"],
                "date": "2025-02-12",
                "thread_id": "thread-002",
            }
        ]
    }

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_response
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(query="support ticket")

    assert len(results) == 1
    assert results[0].text == "Customer is unable to log in."
    assert results[0].metadata["subject"] == "Support ticket #42"


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_uses_body_fallback(mock_igpt_class):
    """Test load_data() falls back to 'body' when 'content' is absent."""
    mock_results = [
        {
            "id": "msg-4",
            "subject": "Fallback test",
            "body": "This message uses the body field.",
            "from": "a@example.com",
            "to": ["b@example.com"],
            "date": "2025-01-01",
            "thread_id": "thread-003",
        }
    ]

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_results
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(query="fallback")

    assert results[0].text == "This message uses the body field."


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_default_max_results(mock_igpt_class):
    """Test load_data() passes default max_results of 50."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = []
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    reader.load_data(query="weekly summary")

    mock_client.recall.search.assert_called_once_with(
        query="weekly summary",
        date_from=None,
        date_to=None,
        max_results=50,
    )


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_empty_response(mock_igpt_class):
    """Test load_data() returns empty list when API returns nothing."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = []
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(query="nothing here")

    assert results == []


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_none_response(mock_igpt_class):
    """Test load_data() returns empty list when API returns None."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = None
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(query="nothing")

    assert results == []


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_non_dict_items(mock_igpt_class):
    """Test load_data() handles non-dict items in results gracefully."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = ["raw string result"]
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(query="edge case")

    assert len(results) == 1
    assert results[0].text == "raw string result"
    assert results[0].metadata["source"] == "igpt_email"


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_error_response(mock_igpt_class):
    """Test load_data() raises ValueError on API error response."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = {"error": "auth"}
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="bad-key", user="test-user")
    with pytest.raises(ValueError, match="iGPT API error: auth"):
        reader.load_data(query="test")


@patch("llama_index.readers.igpt_email.base.IGPT")
def test_load_data_item_without_content_or_body(mock_igpt_class):
    """Test load_data() falls back to json.dumps when item has no content or body."""
    mock_results = [
        {
            "id": "msg-5",
            "subject": "Metadata only",
            "from": "a@example.com",
        }
    ]

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_results
    mock_igpt_class.return_value = mock_client

    reader = IGPTEmailReader(api_key="test-key", user="test-user")
    results = reader.load_data(query="metadata only")

    assert len(results) == 1
    assert results[0].text == json.dumps(mock_results[0])
