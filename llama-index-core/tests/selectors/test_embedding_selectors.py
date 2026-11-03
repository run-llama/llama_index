from typing import List

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.selectors.embedding_selectors import EmbeddingSingleSelector
from llama_index.core.tools.types import ToolMetadata


class _CountingEmbedding(BaseEmbedding):
    single_text_calls: int = 0
    batch_text_calls: int = 0
    async_single_text_calls: int = 0
    async_batch_text_calls: int = 0

    def _embed(self, text: str) -> Embedding:
        vec = [0.0, 0.0, 0.0]
        if "finance" in text:
            vec[0] = 1.0
        elif "weather" in text:
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        return vec

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        self.single_text_calls += 1
        return self._embed(text)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        self.async_single_text_calls += 1
        return self._embed(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        self.batch_text_calls += 1
        return [self._embed(t) for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        self.async_batch_text_calls += 1
        return [self._embed(t) for t in texts]


def test_embedding_single_selector_uses_batch_embeddings() -> None:
    embed_model = _CountingEmbedding(embed_batch_size=10)
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)
    choices = [
        ToolMetadata(name="tool_finance", description="finance reports"),
        ToolMetadata(name="tool_weather", description="weather forecast"),
        ToolMetadata(name="tool_sports", description="sports news"),
    ]

    result = selector.select(choices, "check weather today")
    assert len(result.selections) == 1
    assert result.selections[0].index == 1
    assert "tool_weather" in result.selections[0].reason
    assert embed_model.single_text_calls == 0
    assert embed_model.batch_text_calls == 1


@pytest.mark.asyncio
async def test_embedding_single_selector_async_uses_batch_embeddings() -> None:
    embed_model = _CountingEmbedding(embed_batch_size=2)
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)
    choices = [
        ToolMetadata(name="tool_finance", description="finance reports"),
        ToolMetadata(name="tool_weather", description="weather forecast"),
        ToolMetadata(name="tool_sports", description="sports news"),
    ]

    result = await selector.aselect(choices, "latest sports scores")
    assert len(result.selections) == 1
    assert result.selections[0].index == 2
    assert "tool_sports" in result.selections[0].reason
    assert embed_model.async_single_text_calls == 0
    assert embed_model.async_batch_text_calls == 2
