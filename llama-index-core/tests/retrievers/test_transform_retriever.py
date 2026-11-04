from threading import get_ident
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.query.query_transform.base import IdentityQueryTransform
from llama_index.core.retrievers.transform_retriever import TransformRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_transform_retriever_dispatch(
    use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = QueryBundle(query_str="original")
    transformed = QueryBundle(
        query_str="rewritten", custom_embedding_strs=["embedding"]
    )
    metadata = {"index_summary": "summary"}
    threads = []
    caller_thread = get_ident()

    def transform(query: QueryBundle, metadata: dict) -> QueryBundle:
        threads.append(get_ident())
        assert query is original
        assert metadata == {"index_summary": "summary"}
        return transformed

    query_transform = IdentityQueryTransform()
    run = MagicMock(side_effect=transform)
    monkeypatch.setattr(query_transform, "run", run)
    results = [NodeWithScore(node=TextNode(text="result"), score=0.5)]
    backend = MagicMock(spec=BaseRetriever)
    backend.aretrieve = AsyncMock(return_value=results)
    backend.retrieve.return_value = results
    if use_async:
        backend.retrieve.side_effect = RuntimeError("sync backend unavailable")
    retriever = TransformRetriever(
        backend, query_transform, transform_metadata=metadata
    )

    returned = (
        await retriever.aretrieve(original)
        if use_async
        else retriever.retrieve(original)
    )

    assert returned == results
    run.assert_called_once_with(original, metadata=metadata)
    if use_async:
        backend.aretrieve.assert_awaited_once_with(transformed)
        backend.retrieve.assert_not_called()
        assert len(threads) == 1
        assert threads[0] != caller_thread
    else:
        backend.retrieve.assert_called_once_with(transformed)
        backend.aretrieve.assert_not_called()
        assert threads == [caller_thread]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_transform_failure_does_not_call_backend(
    use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("transformation failed")
    transform = IdentityQueryTransform()
    monkeypatch.setattr(transform, "run", MagicMock(side_effect=error))
    backend = MagicMock(spec=BaseRetriever)
    retriever = TransformRetriever(backend, transform)

    with pytest.raises(ValueError, match="transformation failed") as caught:
        if use_async:
            await retriever.aretrieve("query")
        else:
            retriever.retrieve("query")

    assert caught.value is error
    backend.retrieve.assert_not_called()
    backend.aretrieve.assert_not_called()
