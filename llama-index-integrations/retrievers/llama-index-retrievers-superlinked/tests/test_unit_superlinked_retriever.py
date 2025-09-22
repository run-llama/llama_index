"""Unit tests for SuperlinkedRetriever (LlamaIndex)."""

import pytest
from typing import Any
from unittest.mock import Mock

from llama_index.retrievers.superlinked import SuperlinkedRetriever


# Patch superlinked types before importing the retriever to satisfy validators
class MockApp:
    pass


class MockQuery:
    pass


@pytest.fixture(autouse=True)
def _patch_superlinked_modules(monkeypatch: Any) -> None:
    import sys

    mock_app_module = Mock()
    mock_query_module = Mock()

    mock_app_module.App = MockApp
    mock_query_module.QueryDescriptor = MockQuery

    sys.modules["superlinked.framework.dsl.app.app"] = mock_app_module
    sys.modules["superlinked.framework.dsl.query.query_descriptor"] = mock_query_module


def test_retriever_validate_and_retrieve_success() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
        top_k=4,
    )

    # Build fake Superlinked response
    mock_entry1 = Mock()
    mock_entry1.id = "1"
    mock_entry1.fields = {"text": "Paris is beautiful.", "city": "Paris"}
    mock_entry1.metadata = Mock(score=0.9)

    mock_entry2 = Mock()
    mock_entry2.id = "2"
    mock_entry2.fields = {"text": "Rome has the Colosseum.", "city": "Rome"}
    mock_entry2.metadata = Mock(score=0.8)

    mock_result = Mock()
    mock_result.entries = [mock_entry1, mock_entry2]

    retriever.sl_client.query = Mock(return_value=mock_result)

    nodes = retriever.retrieve("cities")
    assert len(nodes) == 2
    assert nodes[0].node.text in {"Paris is beautiful.", "Rome has the Colosseum."}
    # metadata should include id and city
    md = nodes[0].node.metadata
    assert "id" in md and "city" in md
    # scores should be propagated
    scores = sorted([n.score for n in nodes], reverse=True)
    assert scores == [0.9, 0.8]


def test_retriever_respects_k() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
        top_k=1,
    )

    mock_entry = Mock()
    mock_entry.id = "1"
    mock_entry.fields = {"text": "A", "x": 1}

    mock_entry2 = Mock()
    mock_entry2.id = "2"
    mock_entry2.fields = {"text": "B", "x": 2}

    mock_result = Mock()
    mock_result.entries = [mock_entry, mock_entry2]
    retriever.sl_client.query = Mock(return_value=mock_result)

    nodes = retriever.retrieve("q")
    assert len(nodes) == 1


def test_retriever_metadata_fields_subset() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
        metadata_fields=["city"],
    )

    mock_entry = Mock()
    mock_entry.id = "1"
    mock_entry.fields = {"text": "A", "city": "Paris", "drop": True}

    mock_result = Mock(entries=[mock_entry])
    retriever.sl_client.query = Mock(return_value=mock_result)

    nodes = retriever.retrieve("q")
    assert nodes[0].node.metadata == {"id": "1", "city": "Paris"}


def test_retriever_missing_page_content_skips() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
    )

    mock_entry = Mock()
    mock_entry.id = "1"
    mock_entry.fields = {"not_text": "oops"}

    mock_result = Mock(entries=[mock_entry])
    retriever.sl_client.query = Mock(return_value=mock_result)

    nodes = retriever.retrieve("q")
    assert nodes == []


def test_retriever_query_exception_returns_empty() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
    )

    retriever.sl_client.query = Mock(side_effect=Exception("failure"))
    nodes = retriever.retrieve("q")
    assert nodes == []


def test_query_text_param_is_used() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
        query_text_param="search_term",
    )

    mock_result = Mock(entries=[])
    retriever.sl_client.query = Mock(return_value=mock_result)

    retriever.retrieve("hello")
    retriever.sl_client.query.assert_called_once()
    kwargs = retriever.sl_client.query.call_args.kwargs
    assert kwargs["query_descriptor"] is retriever.sl_query
    assert kwargs["search_term"] == "hello"
