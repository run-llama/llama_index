"""Unit tests for AzureCosmosDBNoSqlVectorSearch.delete()."""

from __future__ import annotations

from typing import Any, Dict, List

from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch

_INDEXING_POLICY = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}
_VECTOR_EMBEDDING_POLICY = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3,
        }
    ]
}


class _RecordingContainer:
    """Container double that records the query and parameters it was asked to run."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = items
        self.queries: List[Dict[str, Any]] = []
        self.deleted: List[Dict[str, Any]] = []

    def query_items(
        self, query: str, parameters: List[Dict[str, Any]] | None = None, **kwargs: Any
    ):
        self.queries.append({"query": query, "parameters": parameters or []})
        return iter(self._items)

    def delete_item(self, item: str, partition_key: str) -> None:
        self.deleted.append({"id": item, "partition_key": partition_key})


class _RecordingDatabase:
    def __init__(self, container: _RecordingContainer) -> None:
        self._container = container

    def create_container_if_not_exists(self, **kwargs: Any) -> _RecordingContainer:
        return self._container


class _RecordingClient:
    def __init__(self, container: _RecordingContainer) -> None:
        self._database = _RecordingDatabase(container)

    def create_database_if_not_exists(self, **kwargs: Any) -> _RecordingDatabase:
        return self._database


def _store(container: _RecordingContainer) -> AzureCosmosDBNoSqlVectorSearch:
    return AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=_RecordingClient(container),
        vector_embedding_policy=_VECTOR_EMBEDDING_POLICY,
        indexing_policy=_INDEXING_POLICY,
        cosmos_container_properties={"partition_key": "/id"},
        create_container=True,
        cosmos_database_properties={},
    )


def test_delete_parameterizes_ref_doc_id() -> None:
    """A ref_doc_id must never be interpolated into the Cosmos SQL text."""
    container = _RecordingContainer(items=[])
    store = _store(container)

    store.delete(ref_doc_id="doc-1")

    assert len(container.queries) == 1
    query = container.queries[0]
    assert query["parameters"] == [{"name": "@ref_doc_id", "value": "doc-1"}]
    assert "@ref_doc_id" in query["query"]
    assert "doc-1" not in query["query"]


def test_delete_does_not_treat_ref_doc_id_as_query_syntax() -> None:
    """An injected payload must stay a value, so only the intended item is deleted."""
    container = _RecordingContainer(
        items=[
            {"id": "doc-1", "partitionKey": "doc-1"},
            {"id": "doc-2", "partitionKey": "doc-2"},
        ]
    )
    store = _store(container)
    payload = "nonexistent' OR 1=1 --"

    store.delete(ref_doc_id=payload)

    query = container.queries[0]
    assert payload not in query["query"]
    assert query["parameters"] == [{"name": "@ref_doc_id", "value": payload}]
    # whatever the server returns is deleted, and it is deleted by partition key
    assert [item["id"] for item in container.deleted] == ["doc-1", "doc-2"]


def test_delete_uses_the_configured_metadata_key() -> None:
    container = _RecordingContainer(items=[])
    store = AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=_RecordingClient(container),
        vector_embedding_policy=_VECTOR_EMBEDDING_POLICY,
        indexing_policy=_INDEXING_POLICY,
        cosmos_container_properties={"partition_key": "/id"},
        create_container=True,
        cosmos_database_properties={},
        metadata_key="meta",
    )

    store.delete(ref_doc_id="doc-1")

    assert "c.meta.ref_doc_id" in container.queries[0]["query"]
