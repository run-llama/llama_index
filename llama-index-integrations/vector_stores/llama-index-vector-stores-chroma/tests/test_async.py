"""Tests for non-blocking Chroma queries."""

import threading
from typing import Any
from unittest.mock import Mock

import chromadb
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.chroma import ChromaVectorStore


@pytest.mark.asyncio
async def test_aquery_runs_query_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChromaVectorStore(chroma_collection=Mock())
    query = VectorStoreQuery(query_embedding=[1.0, 0.0])
    expected = VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
    loop_thread = threading.get_ident()
    query_threads = []

    def query_sync(*args: Any, **kwargs: Any) -> VectorStoreQueryResult:
        query_threads.append(threading.get_ident())
        return expected

    query_mock = Mock(side_effect=query_sync)
    monkeypatch.setattr(ChromaVectorStore, "query", query_mock)

    result = await store.aquery(query, where={"product": "aws"})

    assert result is expected
    query_mock.assert_called_once_with(query, where={"product": "aws"})
    assert len(query_threads) == 1
    assert query_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_aquery_propagates_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ChromaVectorStore(chroma_collection=Mock())
    error = ValueError("invalid query")
    monkeypatch.setattr(ChromaVectorStore, "query", Mock(side_effect=error))

    with pytest.raises(ValueError, match="invalid query") as exc_info:
        await store.aquery(VectorStoreQuery())

    assert exc_info.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["default", "filter_only", "mmr"])
async def test_aquery_matches_sync_query(mode: str) -> None:
    client = chromadb.EphemeralClient()
    collection = client.create_collection("async_query_" + mode)
    try:
        store = ChromaVectorStore(chroma_collection=collection)
        store.add(
            [
                TextNode(
                    id_="aws",
                    text="AWS rule",
                    embedding=[1.0, 0.0],
                    metadata={"product": "aws"},
                ),
                TextNode(
                    id_="windows",
                    text="Windows rule",
                    embedding=[0.0, 1.0],
                    metadata={"product": "windows"},
                ),
            ]
        )
        query = VectorStoreQuery(
            query_embedding=None if mode == "filter_only" else [1.0, 0.0],
            similarity_top_k=1,
            mode=VectorStoreQueryMode.MMR
            if mode == "mmr"
            else VectorStoreQueryMode.DEFAULT,
        )
        expected = store.query(query, where={"product": "aws"})
        result = await store.aquery(query, where={"product": "aws"})

        assert result.ids == expected.ids == ["aws"]
        assert result.similarities == expected.similarities
        assert result.nodes is not None
        assert expected.nodes is not None
        assert [node.model_dump() for node in result.nodes] == [
            node.model_dump() for node in expected.nodes
        ]
    finally:
        client.delete_collection(collection.name)
