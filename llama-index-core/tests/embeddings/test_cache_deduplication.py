from typing import List, Optional

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.storage.kvstore import SimpleKVStore


class RecordingEmbedding(MockEmbedding):
    requested_texts: List[str] = Field(default_factory=list)

    def _get_text_embedding(self, text: str) -> List[float]:
        self.requested_texts.append(text)
        return [float(len(text))]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


@pytest.mark.parametrize("pre_cached", [False, True])
def test_cached_batch_embeds_each_missing_text_once(pre_cached: bool) -> None:
    cache = SimpleKVStore()
    if pre_cached:
        cache.put("warm", {"cached": [42.0]}, collection="embeddings")
    model = RecordingEmbedding(embed_dim=1, embeddings_cache=cache)
    texts = ["alpha", "warm", "alpha", "beta", "beta", "warm"]
    warm_embedding = [42.0] if pre_cached else [4.0]
    expected = [[5.0], warm_embedding, [5.0], [4.0], [4.0], warm_embedding]
    requested = ["alpha", "beta"] if pre_cached else ["alpha", "warm", "beta"]

    assert model.get_text_embedding_batch(texts) == expected
    assert model.requested_texts == requested
    assert model.get_text_embedding_batch(texts) == expected
    assert model.requested_texts == requested


@pytest.mark.asyncio
@pytest.mark.parametrize("num_workers", [None, 2])
@pytest.mark.parametrize("pre_cached", [False, True])
async def test_async_cached_batch_embeds_each_missing_text_once(
    num_workers: Optional[int], pre_cached: bool
) -> None:
    cache = SimpleKVStore()
    if pre_cached:
        cache.put("warm", {"cached": [42.0]}, collection="embeddings")
    model = RecordingEmbedding(
        embed_dim=1, embeddings_cache=cache, num_workers=num_workers
    )
    texts = ["alpha", "warm", "alpha", "beta", "beta", "warm"]
    warm_embedding = [42.0] if pre_cached else [4.0]
    expected = [[5.0], warm_embedding, [5.0], [4.0], [4.0], warm_embedding]
    requested = ["alpha", "beta"] if pre_cached else ["alpha", "warm", "beta"]

    assert await model.aget_text_embedding_batch(texts) == expected
    assert model.requested_texts == requested
    assert await model.aget_text_embedding_batch(texts) == expected
    assert model.requested_texts == requested


@pytest.mark.asyncio
async def test_batches_without_cache_keep_repeated_inputs() -> None:
    model = RecordingEmbedding(embed_dim=1)
    texts = ["alpha", "beta", "alpha"]
    expected = [[5.0], [4.0], [5.0]]

    assert model.get_text_embedding_batch(texts) == expected
    assert model.requested_texts == texts
    model.requested_texts.clear()
    assert await model.aget_text_embedding_batch(texts) == expected
    assert model.requested_texts == texts


@pytest.mark.asyncio
async def test_empty_cached_batches_do_not_embed() -> None:
    model = RecordingEmbedding(embed_dim=1, embeddings_cache=SimpleKVStore())

    assert model.get_text_embedding_batch([]) == []
    assert await model.aget_text_embedding_batch([]) == []
    assert model.requested_texts == []
