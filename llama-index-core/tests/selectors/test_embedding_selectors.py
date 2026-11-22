from typing import List

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors.embedding_selectors import EmbeddingSingleSelector


class _CountingEmbedding(BaseEmbedding):
    batch_calls: int = 0
    async_batch_calls: int = 0
    single_calls: int = 0
    async_single_calls: int = 0
    query_calls: int = 0
    async_query_calls: int = 0
    last_batch_texts: List[str] = Field(default_factory=list)
    last_async_batch_texts: List[str] = Field(default_factory=list)

    def _get_query_embedding(self, query: str) -> List[float]:
        self.query_calls += 1
        return [1.0, 0.0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        self.async_query_calls += 1
        return [0.0, 1.0]

    def _get_text_embedding(self, text: str) -> List[float]:
        self.single_calls += 1
        return [1.0, 0.0] if text == "apple" else [0.0, 1.0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        self.async_single_calls += 1
        return [1.0, 0.0] if text == "apple" else [0.0, 1.0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.batch_calls += 1
        self.last_batch_texts = list(texts)
        return [[1.0, 0.0] if t == "apple" else [0.0, 1.0] for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.async_batch_calls += 1
        self.last_async_batch_texts = list(texts)
        return [[1.0, 0.0] if t == "apple" else [0.0, 1.0] for t in texts]


def test_embedding_single_selector_uses_batch_embeddings() -> None:
    embed_model = _CountingEmbedding()
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)

    result = selector.select(choices=["apple", "pear", "peach"], query="pick apple")

    assert result.ind == 0
    assert embed_model.last_batch_texts == ["apple", "pear", "peach"]
    assert embed_model.batch_calls == 1
    assert embed_model.single_calls == 0
    assert embed_model.query_calls == 1


@pytest.mark.asyncio
async def test_embedding_single_selector_async_uses_batch_embeddings() -> None:
    embed_model = _CountingEmbedding()
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)

    result = await selector.aselect(choices=["apple", "pear"], query="pick pear")

    assert result.ind == 1
    assert embed_model.last_async_batch_texts == ["apple", "pear"]
    assert embed_model.async_batch_calls == 1
    assert embed_model.async_single_calls == 0
    assert embed_model.async_query_calls == 1


@pytest.mark.asyncio
async def test_embedding_single_selector_reuses_precomputed_query_embedding() -> None:
    embed_model = _CountingEmbedding()
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)

    query_bundle = QueryBundle(query_str="precomputed", embedding=[0.0, 1.0])
    sync_result = selector.select(choices=["apple", "pear"], query=query_bundle)
    async_result = await selector.aselect(choices=["apple", "pear"], query=query_bundle)

    assert sync_result.ind == 1
    assert async_result.ind == 1
    assert embed_model.query_calls == 0
    assert embed_model.async_query_calls == 0


@pytest.mark.asyncio
async def test_embedding_single_selector_validates_empty_choices() -> None:
    embed_model = _CountingEmbedding()
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)

    with pytest.raises(ValueError, match="Choices must be non-empty"):
        selector.select(choices=[], query="empty")

    with pytest.raises(ValueError, match="Choices must be non-empty"):
        await selector.aselect(choices=[], query="empty")
    assert embed_model.query_calls == 0
    assert embed_model.async_query_calls == 0
    assert embed_model.batch_calls == 0
    assert embed_model.async_batch_calls == 0
