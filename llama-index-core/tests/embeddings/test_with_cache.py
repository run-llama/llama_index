import pytest
from llama_index.core import MockEmbedding
from llama_index.core.storage.kvstore import SimpleKVStore
from unittest.mock import patch

expected_embedding = [0.5, 0.5, 0.5, 0.5]


def test_cache_recomputes_legacy_entries():
    cache = SimpleKVStore()
    legacy_entry = {"old": [9.0, 9.0]}
    cache.put(key="Paris", val=legacy_entry, collection="embeddings")
    embed_model = MockEmbedding(embed_dim=2, embeddings_cache=cache)

    with (
        patch.object(embed_model, "_get_query_embedding", return_value=[1.0, 0.0]),
        patch.object(embed_model, "_get_text_embedding", return_value=[0.0, 1.0]),
    ):
        assert embed_model.get_query_embedding("Paris") == [1.0, 0.0]
        assert embed_model.get_text_embedding("Paris") == [0.0, 1.0]

    assert cache.get(key="Paris", collection="embeddings") == legacy_entry


@pytest.mark.parametrize("query_first", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_cache_preserves_query_and_text_embeddings(query_first, batch):
    embed_model = MockEmbedding(embed_dim=2, embeddings_cache=SimpleKVStore())
    query_embedding = [1.0, 0.0]
    text_embedding = [0.0, 1.0]
    text_method = (
        embed_model.get_text_embedding_batch
        if batch
        else embed_model.get_text_embedding
    )
    text_input = ["Paris"] if batch else "Paris"
    expected_text = [text_embedding] if batch else text_embedding
    provider_method = "_get_text_embeddings" if batch else "_get_text_embedding"

    with (
        patch.object(
            embed_model, "_get_query_embedding", return_value=query_embedding
        ) as query_provider,
        patch.object(
            embed_model, provider_method, return_value=expected_text
        ) as text_provider,
    ):
        if query_first:
            assert embed_model.get_query_embedding("Paris") == query_embedding
        assert text_method(text_input) == expected_text
        assert embed_model.get_query_embedding("Paris") == query_embedding
        assert text_method(text_input) == expected_text
        assert embed_model.get_query_embedding("Paris") == query_embedding

    query_provider.assert_called_once_with("Paris")
    text_provider.assert_called_once_with(text_input)


@pytest.mark.asyncio
@pytest.mark.parametrize("query_first", [False, True])
@pytest.mark.parametrize("batch", [False, True])
async def test_async_cache_preserves_query_and_text_embeddings(query_first, batch):
    embed_model = MockEmbedding(embed_dim=2, embeddings_cache=SimpleKVStore())
    query_embedding = [1.0, 0.0]
    text_embedding = [0.0, 1.0]
    text_method = (
        embed_model.aget_text_embedding_batch
        if batch
        else embed_model.aget_text_embedding
    )
    text_input = ["Paris"] if batch else "Paris"
    expected_text = [text_embedding] if batch else text_embedding
    provider_method = "_aget_text_embeddings" if batch else "_aget_text_embedding"

    with (
        patch.object(
            embed_model, "_aget_query_embedding", return_value=query_embedding
        ) as query_provider,
        patch.object(
            embed_model, provider_method, return_value=expected_text
        ) as text_provider,
    ):
        if query_first:
            assert await embed_model.aget_query_embedding("Paris") == query_embedding
        assert await text_method(text_input) == expected_text
        assert await embed_model.aget_query_embedding("Paris") == query_embedding
        assert await text_method(text_input) == expected_text
        assert await embed_model.aget_query_embedding("Paris") == query_embedding

    query_provider.assert_awaited_once_with("Paris")
    text_provider.assert_awaited_once_with(text_input)


# Create unique embeddings for each text to verify order
def custom_embeddings(texts):
    return [[float(ord(c)) for c in text[-4:]] for text in texts]


def test_sync_get_with_cache():
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    text = "Hello"

    text_embedding = embed_model.get_text_embedding(text)
    assert text_embedding == expected_embedding
    assert embeddings_cache.get(key="Hello", collection="text_embeddings") is not None

    embd_dict = embeddings_cache.get(key="Hello", collection="text_embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embedding


def test_sync_get_batch_with_cache():
    """Test mixed scenario with some cached and some new inputs."""
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    texts = ["Cached1", "Miss1", "Cached2", "Miss2"]

    # Pre-cache
    embed_model.embeddings_cache.put(
        key="Cached1",
        val={"uuid1": [104.0, 101.0, 100.0, 49.0]},
        collection="text_embeddings",
    )
    embed_model.embeddings_cache.put(
        key="Cached2",
        val={"uuid3": [104.0, 101.0, 100.0, 50.0]},
        collection="text_embeddings",
    )

    with patch.object(
        embed_model,
        "_get_text_embeddings",
        side_effect=custom_embeddings,
    ) as mock_get_embeddings:
        text_embeddings = embed_model.get_text_embedding_batch(texts)

        expected_embeddings = [
            [104.0, 101.0, 100.0, 49.0],  # Cached1
            [105.0, 115.0, 115.0, 49.0],  # Miss1 (first in batch)
            [104.0, 101.0, 100.0, 50.0],  # Cached2
            [105.0, 115.0, 115.0, 50.0],  # Miss2 (second in batch)
        ]
        assert text_embeddings == expected_embeddings

        assert mock_get_embeddings.call_count == 1

        # Check cache
        for i, text in enumerate(texts):
            embd_dict = embeddings_cache.get(key=text, collection="text_embeddings")

            first_key = next(iter(embd_dict.keys()))
            assert embd_dict[first_key] == expected_embeddings[i]


@pytest.mark.asyncio
async def test_async_get_with_cache():
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    text = "Hello"

    text_embedding = await embed_model.aget_text_embedding(text)
    assert text_embedding == expected_embedding
    assert embeddings_cache.get(key="Hello", collection="text_embeddings") is not None

    embd_dict = embeddings_cache.get(key="Hello", collection="text_embeddings")
    first_key = next(iter(embd_dict.keys()))
    assert embd_dict[first_key] == expected_embedding


@pytest.mark.asyncio
async def test_async_get_batch_with_cache():
    """Test mixed scenario with some cached and some new inputs."""
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    texts = ["Cached1", "Miss1", "Cached2", "Miss2"]

    # Pre-cache
    embed_model.embeddings_cache.put(
        key="Cached1",
        val={"uuid1": [104.0, 101.0, 100.0, 49.0]},
        collection="text_embeddings",
    )
    embed_model.embeddings_cache.put(
        key="Cached2",
        val={"uuid3": [104.0, 101.0, 100.0, 50.0]},
        collection="text_embeddings",
    )

    with patch.object(
        embed_model,
        "_aget_text_embeddings",
        side_effect=custom_embeddings,
    ) as mock_get_embeddings:
        text_embeddings = await embed_model.aget_text_embedding_batch(texts)

        expected_embeddings = [
            [104.0, 101.0, 100.0, 49.0],  # Cached1
            [105.0, 115.0, 115.0, 49.0],  # Miss1 (first in batch)
            [104.0, 101.0, 100.0, 50.0],  # Cached2
            [105.0, 115.0, 115.0, 50.0],  # Miss2 (second in batch)
        ]
        assert text_embeddings == expected_embeddings

        assert mock_get_embeddings.call_count == 1

        # Check cache
        for i, text in enumerate(texts):
            embd_dict = embeddings_cache.get(key=text, collection="text_embeddings")

            first_key = next(iter(embd_dict.keys()))
            assert embd_dict[first_key] == expected_embeddings[i]
