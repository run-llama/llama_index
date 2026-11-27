import asyncio

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.timescalevector import TimescaleVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TimescaleVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def _row(node_id, distance):
    node = TextNode(id_=node_id, text=node_id)
    # (id, metadata, contents, embedding, distance), as returned by
    # timescale_vector's client.search()
    return (node_id, node_to_metadata_dict(node), node_id, [0.0, 0.0, 0.0], distance)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(TimescaleVectorStore, "_create_tables", lambda self: None)
    return TimescaleVectorStore(
        service_url="postgres://placeholder-user@localhost:5432/db",
        table_name="test_table",
        num_dimensions=3,
    )


def test_query_returns_similarities_not_distances(store, monkeypatch):
    rows = [_row("exact", 0.0), _row("close", 0.05), _row("far", 1.9)]
    monkeypatch.setattr(store._sync_client, "search", lambda *a, **kw: rows)

    result = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3)
    )

    assert result.ids == ["exact", "close", "far"]
    # The client returns pgvector's cosine distance (0 = identical),
    # but similarities must be higher-is-better.
    assert result.similarities == pytest.approx([1.0, 0.95, -0.9])


def test_aquery_returns_similarities_not_distances(store, monkeypatch):
    rows = [_row("exact", 0.0), _row("far", 1.9)]

    async def search(*args, **kwargs):
        return rows

    monkeypatch.setattr(store._async_client, "search", search)

    result = asyncio.run(
        store.aquery(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=2)
        )
    )

    assert result.ids == ["exact", "far"]
    assert result.similarities == pytest.approx([1.0, -0.9])


def test_query_without_embedding_keeps_placeholder_score(store, monkeypatch):
    # client.search() returns -1.0 as the distance when there is no embedding
    rows = [_row("a", -1.0)]
    monkeypatch.setattr(store._sync_client, "search", lambda *a, **kw: rows)

    result = store.query(VectorStoreQuery(query_embedding=None, similarity_top_k=1))

    assert result.similarities == [-1.0]
