"""Tests for IGPTEmailToolSpec."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.tools.igpt_email import IGPTEmailToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in IGPTEmailToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert "ask" in IGPTEmailToolSpec.spec_functions
    assert "search" in IGPTEmailToolSpec.spec_functions


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_to_tool_list(mock_igpt_class):
    """Test that to_tool_list() produces valid FunctionTool objects."""
    tool_spec = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    tools = tool_spec.to_tool_list()

    assert len(tools) == 2
    assert all(isinstance(t, FunctionTool) for t in tools)

    tool_names = {t.metadata.name for t in tools}
    assert tool_names == {"ask", "search"}


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_init(mock_igpt_class):
    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    mock_igpt_class.assert_called_once_with(api_key="test-key", user="test-user")
    assert tool.client == mock_igpt_class.return_value


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_ask_returns_document(mock_igpt_class):
    mock_response = {
        "answer": "You have 3 action items from the project meeting.",
        "citations": [{"id": "msg-1", "subject": "Project sync"}],
        "tasks": ["Send proposal", "Schedule follow-up"],
    }

    mock_client = MagicMock()
    mock_client.recall.ask.return_value = mock_response
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.ask("What are my action items from this week?")

    mock_client.recall.ask.assert_called_once_with(
        input="What are my action items from this week?",
        output_format="json",
    )

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].text == json.dumps(mock_response)
    assert results[0].metadata["question"] == "What are my action items from this week?"
    assert results[0].metadata["citations"] == mock_response["citations"]
    assert results[0].metadata["source"] == "igpt_email_ask"


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_ask_with_text_output_format(mock_igpt_class):
    mock_client = MagicMock()
    mock_client.recall.ask.return_value = {"answer": "Some answer", "citations": []}
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    tool.ask("What deadlines are coming up?", output_format="text")

    mock_client.recall.ask.assert_called_once_with(
        input="What deadlines are coming up?",
        output_format="text",
    )


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_ask_with_string_response(mock_igpt_class):
    """Test ask() handles a plain string response gracefully."""
    mock_client = MagicMock()
    mock_client.recall.ask.return_value = "Plain text answer"
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.ask("Any decisions made?")

    assert len(results) == 1
    assert results[0].text == "Plain text answer"
    assert results[0].metadata["citations"] == []


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_returns_documents(mock_igpt_class):
    mock_results = [
        {
            "id": "msg-1",
            "subject": "Q1 Budget Review",
            "content": "Team, please review the attached Q1 budget.",
            "from": "alice@company.com",
            "to": ["bob@company.com"],
            "date": "2025-02-10",
            "thread_id": "thread-abc",
        },
        {
            "id": "msg-2",
            "subject": "Re: Q1 Budget Review",
            "content": "Looks good. Let's approve.",
            "from": "bob@company.com",
            "to": ["alice@company.com"],
            "date": "2025-02-11",
            "thread_id": "thread-abc",
        },
    ]

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_results
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.search("Q1 budget", date_from="2025-02-01", max_results=5)

    mock_client.recall.search.assert_called_once_with(
        query="Q1 budget",
        date_from="2025-02-01",
        date_to=None,
        max_results=5,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

    assert results[0].text == "Team, please review the attached Q1 budget."
    assert results[0].metadata["subject"] == "Q1 Budget Review"
    assert results[0].metadata["from"] == "alice@company.com"
    assert results[0].metadata["thread_id"] == "thread-abc"
    assert results[0].metadata["source"] == "igpt_email_search"

    assert results[1].text == "Looks good. Let's approve."
    assert results[1].metadata["id"] == "msg-2"


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_with_results_wrapper(mock_igpt_class):
    """Test search() handles a dict response with a 'results' key."""
    mock_response = {
        "results": [
            {
                "id": "msg-1",
                "subject": "Deal update",
                "content": "The deal is closed.",
                "from": "sales@company.com",
                "to": ["ceo@company.com"],
                "date": "2025-02-14",
                "thread_id": "thread-xyz",
            }
        ]
    }

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_response
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.search("deal update")

    assert len(results) == 1
    assert results[0].text == "The deal is closed."
    assert results[0].metadata["subject"] == "Deal update"


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_empty_response(mock_igpt_class):
    """Test search() returns empty list when API returns nothing."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = []
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.search("nonexistent topic")

    assert results == []


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_none_response(mock_igpt_class):
    """Test search() returns empty list when API returns None."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = None
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.search("nothing here")

    assert results == []


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_default_parameters(mock_igpt_class):
    """Test search() uses correct defaults when optional args are omitted."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = []
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    tool.search("onboarding")

    mock_client.recall.search.assert_called_once_with(
        query="onboarding",
        date_from=None,
        date_to=None,
        max_results=10,
    )


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_ask_error_response(mock_igpt_class):
    """Test ask() raises ValueError on API error response."""
    mock_client = MagicMock()
    mock_client.recall.ask.return_value = {"error": "auth"}
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="bad-key", user="test-user")
    with pytest.raises(ValueError, match="iGPT API error: auth"):
        tool.ask("test question")


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_error_response(mock_igpt_class):
    """Test search() raises ValueError on API error response."""
    mock_client = MagicMock()
    mock_client.recall.search.return_value = {"error": "params"}
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    with pytest.raises(ValueError, match="iGPT API error: params"):
        tool.search("bad query")


@patch("llama_index.tools.igpt_email.base.IGPT")
def test_search_item_without_content_or_body(mock_igpt_class):
    """Test search() falls back to json.dumps when item has no content or body."""
    mock_results = [
        {
            "id": "msg-1",
            "subject": "Metadata only",
            "from": "a@example.com",
        }
    ]

    mock_client = MagicMock()
    mock_client.recall.search.return_value = mock_results
    mock_igpt_class.return_value = mock_client

    tool = IGPTEmailToolSpec(api_key="test-key", user="test-user")
    results = tool.search("metadata only")

    assert len(results) == 1
    assert results[0].text == json.dumps(mock_results[0])
