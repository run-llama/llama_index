import asyncio
from typing import Any, List

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.embeddings.base_sparse import (
    BaseSparseEmbedding,
    SparseEmbedding,
)
from llama_index.core.bridge.pydantic import Field


class _RecordingEmbedding(BaseEmbedding):
    """Dense embedding whose per-text calls can fail and report cancellation."""

    failing_text: str = "bad"
    started: List[str] = Field(default_factory=list)
    cancelled: List[str] = Field(default_factory=list)
    finished: List[str] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "recording"

    def _get_text_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    def _get_query_embedding(self, query: str) -> List[float]:
        raise NotImplementedError

    async def _aget_query_embedding(self, query: str) -> List[float]:
        raise NotImplementedError

    async def _aget_text_embedding(self, text: str) -> List[float]:
        self.started.append(text)
        try:
            if text == self.failing_text:
                await asyncio.sleep(0)
                raise ValueError(f"embedding rejected: {text}")
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        self.finished.append(text)
        return [0.1, 0.2, 0.3]


class _RecordingSparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding with the same failure and cancellation bookkeeping."""

    failing_text: str = "bad"
    started: List[str] = Field(default_factory=list)
    cancelled: List[str] = Field(default_factory=list)

    def _get_text_embedding(self, text: str) -> SparseEmbedding:
        raise NotImplementedError

    def _get_query_embedding(self, query: str) -> SparseEmbedding:
        raise NotImplementedError

    async def _aget_query_embedding(self, query: str) -> SparseEmbedding:
        raise NotImplementedError

    async def _aget_text_embedding(self, text: str) -> SparseEmbedding:
        self.started.append(text)
        try:
            if text == self.failing_text:
                await asyncio.sleep(0)
                raise ValueError(f"embedding rejected: {text}")
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        return {1: 1.0}


@pytest.mark.asyncio
async def test_batch_embeddings_cancel_siblings_on_failure():
    embed_model = _RecordingEmbedding()

    with pytest.raises(ValueError):
        await embed_model._aget_text_embeddings(["bad", "slow-1", "slow-2"])

    assert sorted(embed_model.started) == ["bad", "slow-1", "slow-2"]
    assert sorted(embed_model.cancelled) == ["slow-1", "slow-2"]
    assert embed_model.finished == []


@pytest.mark.asyncio
async def test_aget_text_embedding_batch_cancels_sibling_batches_on_failure():
    embed_model = _RecordingEmbedding(embed_batch_size=1)

    with pytest.raises(ValueError):
        await embed_model.aget_text_embedding_batch(["bad", "slow-1", "slow-2"])

    assert sorted(embed_model.started) == ["bad", "slow-1", "slow-2"]
    assert sorted(embed_model.cancelled) == ["slow-1", "slow-2"]


@pytest.mark.asyncio
async def test_aget_text_embedding_batch_success_keeps_order():
    embed_model = _RecordingEmbedding(embed_batch_size=1)

    embeddings = await embed_model.aget_text_embedding_batch(["a", "b", "c"])

    assert len(embeddings) == 3
    assert embed_model.cancelled == []
    assert sorted(embed_model.finished) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_sparse_batch_embeddings_cancel_siblings_on_failure():
    embed_model = _RecordingSparseEmbedding()

    with pytest.raises(ValueError):
        await embed_model._aget_text_embeddings(["bad", "slow-1", "slow-2"])

    assert sorted(embed_model.started) == ["bad", "slow-1", "slow-2"]
    assert sorted(embed_model.cancelled) == ["slow-1", "slow-2"]


@pytest.mark.asyncio
async def test_sparse_aget_text_embedding_batch_cancels_sibling_batches_on_failure():
    embed_model = _RecordingSparseEmbedding(embed_batch_size=1)

    with pytest.raises(ValueError):
        await embed_model.aget_text_embedding_batch(["bad", "slow-1", "slow-2"])

    assert sorted(embed_model.started) == ["bad", "slow-1", "slow-2"]
    assert sorted(embed_model.cancelled) == ["slow-1", "slow-2"]


@pytest.mark.asyncio
async def test_gather_cancel_on_error_drains_remaining_tasks():
    from llama_index.core.async_utils import gather_cancel_on_error

    state: dict = {"cancelled": [], "completed": []}

    async def ok(name: str) -> str:
        try:
            await asyncio.sleep(0.2)
            state["completed"].append(name)
            return name
        except asyncio.CancelledError:
            state["cancelled"].append(name)
            raise

    async def boom() -> Any:
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await gather_cancel_on_error([boom(), ok("a"), ok("b")])

    assert state["completed"] == []
    assert sorted(state["cancelled"]) == ["a", "b"]
