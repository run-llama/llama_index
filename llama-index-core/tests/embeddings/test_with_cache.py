import pytest
from llama_index.core import MockEmbedding
from llama_index.core.storage.kvstore import SimpleKVStore
from unittest.mock import patch

expected_embedding = [0.5, 0.5, 0.5, 0.5]
expected_embeddings = [expected_embedding]


def test_sync_get_with_cache():
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    text = "Hello"

    text_embedding = embed_model.get_text_embedding(text)
    assert text_embedding == expected_embedding
    assert embeddings_cache.get(key="Hello", collection="embeddings") is not None

    embd_dict = embeddings_cache.get(key="Hello", collection="embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embedding


def test_sync_get_batch_with_cache():
    """Test mixed scenario with some cached and some new inputs."""
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    texts = ["Cached1", "Miss1", "Cached2", "Miss2"]

    # Pre-cache
    embed_model.get_text_embedding("Cached1")
    embed_model.get_text_embedding("Cached2")

    with patch.object(
        embed_model,
        "_get_text_embeddings",
        wraps=embed_model._get_text_embeddings,
    ) as mock_get_embeddings:
        text_embeddings = embed_model.get_text_embedding_batch(texts)
        assert text_embeddings == expected_embeddings * 4, (
            f"{text_embeddings} != {expected_embeddings * 4}"
        )

        assert mock_get_embeddings.call_count == 1

        # Check cache
        for text in texts:
            embd_dict = embeddings_cache.get(key=text, collection="embeddings")

            first_key = next(iter(embd_dict.keys()))
            assert embd_dict[first_key] == expected_embedding


@pytest.mark.asyncio
async def test_async_get_with_cache():
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    text = "Hello"

    text_embedding = await embed_model.aget_text_embedding(text)
    assert text_embedding == expected_embedding
    assert embeddings_cache.get(key="Hello", collection="embeddings") is not None

    embd_dict = embeddings_cache.get(key="Hello", collection="embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embedding


@pytest.mark.asyncio
async def test_async_get_batch_with_cache():
    """Test mixed scenario with some cached and some new inputs."""
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    texts = ["Cached1", "Miss1", "Cached2", "Miss2"]

    # Pre-cache
    await embed_model.aget_text_embedding("Cached1")
    await embed_model.aget_text_embedding("Cached2")

    with patch.object(
        embed_model,
        "_aget_text_embeddings",
        wraps=embed_model._get_text_embeddings,
    ) as mock_get_embeddings:
        text_embeddings = await embed_model.aget_text_embedding_batch(texts)
        assert text_embeddings == expected_embeddings * 4, (
            f"{text_embeddings} != {expected_embeddings * 4}"
        )

        assert mock_get_embeddings.call_count == 2

        # Check cache
        for text in texts:
            embd_dict = embeddings_cache.get(key=text, collection="embeddings")

            first_key = next(iter(embd_dict.keys()))
            assert embd_dict[first_key] == expected_embedding
