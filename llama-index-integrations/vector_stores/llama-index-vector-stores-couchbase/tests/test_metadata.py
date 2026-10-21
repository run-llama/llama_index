"""Unit tests for metadata returned by Couchbase Search."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore


@pytest.mark.parametrize("metadata_key", ["metadata", "attributes"])
def test_format_metadata_preserves_repeated_prefix(metadata_key: str) -> None:
    store = CouchbaseSearchVectorStore.model_construct()
    store._metadata_key = metadata_key
    fields = {
        f"{metadata_key}.title": "outer title",
        f"{metadata_key}.source.{metadata_key}.title": "source title",
        f"{metadata_key}_extra.{metadata_key}.title": "unrelated title",
        "category": "article",
    }

    assert store._format_metadata(fields) == {
        "title": "outer title",
        f"source.{metadata_key}.title": "source title",
        f"{metadata_key}_extra.{metadata_key}.title": "unrelated title",
        "category": "article",
    }
    assert len(fields) == 4


@pytest.mark.parametrize("scoped_index", [True, False])
def test_query_preserves_distinct_metadata_fields(scoped_index: bool) -> None:
    store = CouchbaseSearchVectorStore.model_construct()
    store._metadata_key = "metadata"
    store._text_key = "text"
    store._embedding_key = "embedding"
    store._index_name = "test-index"
    store._scoped_index = scoped_index
    store._scope = MagicMock()
    store._cluster = MagicMock()
    search_client = store._scope if scoped_index else store._cluster
    search_client.search.return_value.rows.return_value = [
        SimpleNamespace(
            id="node-1",
            score=0.9,
            fields={
                "text": "document text",
                "metadata.title": "outer title",
                "metadata.source.metadata.title": "source title",
            },
        )
    ]

    result = store.query(VectorStoreQuery(query_embedding=[1.0], similarity_top_k=1))

    assert result.ids == ["node-1"]
    assert result.similarities == [0.9]
    assert result.nodes is not None
    assert result.nodes[0].get_content() == "document text"
    assert result.nodes[0].metadata == {
        "title": "outer title",
        "source.metadata.title": "source title",
    }
    search_client.search.assert_called_once()
