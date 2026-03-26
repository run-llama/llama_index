"""Integration-like tests using only mocks to simulate Superlinked behavior."""

import pytest
from typing import Any, List
from unittest.mock import Mock

from llama_index.retrievers.superlinked import SuperlinkedRetriever


# Patch superlinked modules once for all tests
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


def _make_entries(docs: List[dict]) -> list:
    entries = []
    for d in docs:
        m = Mock()
        m.id = d.get("id")
        m.fields = d
        entries.append(m)
    return entries


def test_basic_flow() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
        top_k=4,
    )

    docs = [
        {"id": "1", "text": "Eiffel Tower is in Paris", "category": "landmark"},
        {"id": "2", "text": "Colosseum is in Rome", "category": "landmark"},
        {"id": "3", "text": "Python is a language", "category": "technology"},
    ]
    # attach scores as metadata
    entries = _make_entries(docs)
    for i, e in enumerate(entries):
        e.metadata = Mock(score=1.0 - i * 0.1)
    retriever.sl_client.query = Mock(return_value=Mock(entries=entries))

    nodes = retriever.retrieve("landmarks")
    assert len(nodes) == 3
    assert any("Paris" in n.node.text for n in nodes)
    assert all("id" in n.node.metadata for n in nodes)


def test_k_limit_and_metadata_subset() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
        metadata_fields=["category"],
        top_k=2,
    )

    docs = [
        {"id": "1", "text": "doc1", "category": "A", "x": 1},
        {"id": "2", "text": "doc2", "category": "B", "x": 2},
        {"id": "3", "text": "doc3", "category": "C", "x": 3},
    ]
    entries = _make_entries(docs)
    for i, e in enumerate(entries):
        e.metadata = Mock(score=0.9 - i * 0.1)
    retriever.sl_client.query = Mock(return_value=Mock(entries=entries))

    nodes = retriever.retrieve("q")
    assert len(nodes) == 2
    for n in nodes:
        assert set(n.node.metadata.keys()) == {"id", "category"}
    # verify scores present
    assert all(isinstance(n.score, float) for n in nodes)


def test_error_returns_empty_list() -> None:
    retriever = SuperlinkedRetriever(
        sl_client=MockApp(),
        sl_query=MockQuery(),
        page_content_field="text",
    )

    retriever.sl_client.query = Mock(side_effect=Exception("boom"))

    nodes = retriever.retrieve("q")
    assert nodes == []
