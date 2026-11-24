from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.milvus import MilvusVectorStore


def _hits(embedding_field: str) -> List[List[Dict[str, Any]]]:
    return [
        [
            {
                "id": "1",
                "distance": 0.9,
                "entity": {"text": "a", embedding_field: [1.0, 0.0]},
            },
            {
                "id": "2",
                "distance": 0.1,
                "entity": {"text": "b", embedding_field: [0.0, 1.0]},
            },
        ]
    ]


def _store(embedding_field: str, **kwargs: Any) -> MilvusVectorStore:
    with (
        patch("llama_index.vector_stores.milvus.base.MilvusClient") as client_cls,
        patch("llama_index.vector_stores.milvus.base.AsyncMilvusClient") as aclient_cls,
    ):
        client = MagicMock()
        client.list_collections.return_value = []
        client.search.return_value = _hits(embedding_field)
        client_cls.return_value = client
        aclient = MagicMock()
        aclient.search = AsyncMock(return_value=_hits(embedding_field))
        aclient_cls.return_value = aclient
        return MilvusVectorStore(
            dim=2,
            embedding_field=embedding_field,
            use_async_client=True,
            **kwargs,
        )


def _mmr_query() -> VectorStoreQuery:
    return VectorStoreQuery(
        query_embedding=[1.0, 0.1],
        similarity_top_k=1,
        mode=VectorStoreQueryMode.MMR,
    )


def test_mmr_query_uses_custom_embedding_field() -> None:
    store = _store("vector")
    result = store.query(_mmr_query())

    assert result.ids == ["1"]


@pytest.mark.asyncio
async def test_async_mmr_query_uses_custom_embedding_field() -> None:
    store = _store("vector")
    result = await store.aquery(_mmr_query())

    assert result.ids == ["1"]


def test_mmr_query_requests_embedding_with_limited_output_fields() -> None:
    store = _store("embedding", output_fields=["tag"])
    store.query(_mmr_query())

    output_fields = store.client.search.call_args.kwargs["output_fields"]
    assert "embedding" in output_fields
    assert "tag" in output_fields


@pytest.mark.asyncio
async def test_async_mmr_query_requests_embedding_with_limited_output_fields() -> None:
    store = _store("embedding", output_fields=["tag"])
    await store.aquery(_mmr_query())

    output_fields = store.aclient.search.call_args.kwargs["output_fields"]
    assert "embedding" in output_fields
    assert "tag" in output_fields
