import pytest
from llama_index.core import MockEmbedding
from llama_index.core.storage.kvstore import SimpleKVStore
from typing import List, Any


@pytest.fixture()
def expected_embeddings() -> List[Any]:
    return [[0.5, 0.5, 0.5, 0.5], [[0.5, 0.5, 0.5, 0.5]]]


def test_sync_methods_with_cache(expected_embeddings: List[Any]):
    kwargs_dict = {"embeddings_cache": SimpleKVStore()}
    embed_model = MockEmbedding(embed_dim=4, **kwargs_dict)
    text = "Hello"
    texts = ["Hello"]
    text_embedding = embed_model.get_text_embedding(text)
    assert text_embedding == expected_embeddings[0]
    assert (
        embed_model.embeddings_cache.get(key="Hello", collection="embeddings")
        is not None
    )
    embd_dict = embed_model.embeddings_cache.get(key="Hello", collection="embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embeddings[0]
    text_embeddings = embed_model.get_text_embedding_batch(texts)
    assert text_embeddings == expected_embeddings[1]
    embd_dict = embed_model.embeddings_cache.get(key="Hello", collection="embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embeddings[0]


@pytest.mark.asyncio
async def test_async_methods_with_cache(expected_embeddings: List[Any]):
    kwargs_dict = {"embeddings_cache": SimpleKVStore()}
    embed_model = MockEmbedding(embed_dim=4, **kwargs_dict)
    text = "Hello"
    texts = ["Hello"]
    text_embedding = await embed_model.aget_text_embedding(text)
    assert text_embedding == expected_embeddings[0]
    assert (
        embed_model.embeddings_cache.get(key="Hello", collection="embeddings")
        is not None
    )
    embd_dict = embed_model.embeddings_cache.get(key="Hello", collection="embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embeddings[0]
    text_embeddings = await embed_model.aget_text_embedding_batch(texts)
    assert text_embeddings == expected_embeddings[1]
    embd_dict = embed_model.embeddings_cache.get(key="Hello", collection="embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embeddings[0]
