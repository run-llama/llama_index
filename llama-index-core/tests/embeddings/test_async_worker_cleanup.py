import asyncio
from typing import Any

import pytest

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.sparse_embeddings.mock_sparse_embedding import MockSparseEmbedding


@pytest.mark.asyncio
@pytest.mark.parametrize("show_progress", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
async def test_embedding_workers_clean_up_on_batch_failure(
    monkeypatch: pytest.MonkeyPatch, show_progress: bool, sparse: bool
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned_up = asyncio.Event()
    error = ValueError("batch failed")

    async def embed_batch(self: Any, texts: list[str]) -> list[Any]:
        if texts == ["fail"]:
            await started.wait()
            raise error
        started.set()
        try:
            await release.wait()
            return [self._get_text_embedding(text) for text in texts]
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    model = (
        MockSparseEmbedding(embed_batch_size=1, num_workers=2)
        if sparse
        else MockEmbedding(embed_dim=1, embed_batch_size=1, num_workers=2)
    )
    monkeypatch.setattr(type(model), "_aget_text_embeddings", embed_batch)
    previous_tasks = asyncio.all_tasks()
    try:
        with pytest.raises(ValueError) as exc_info:
            await model.aget_text_embedding_batch(
                ["fail", "active"], show_progress=show_progress
            )
        assert exc_info.value is error
        assert cleaned_up.is_set()
        assert not (asyncio.all_tasks() - previous_tasks)
    finally:
        release.set()
        await asyncio.gather(
            *(asyncio.all_tasks() - previous_tasks), return_exceptions=True
        )
