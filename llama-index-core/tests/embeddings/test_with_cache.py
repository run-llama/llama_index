import re

import pytest
from llama_index.core import MockEmbedding
from llama_index.core.storage.kvstore import SimpleKVStore
from unittest.mock import patch

expected_embedding = [0.5, 0.5, 0.5, 0.5]


# Create unique embeddings for each text to verify order
def custom_embeddings(texts):
    return [[float(ord(c)) for c in text[-4:]] for text in texts]


def test_sync_get_with_cache():
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    text = "Hello"

    text_embedding = embed_model.get_text_embedding(text)
    assert text_embedding == expected_embedding
    assert (
        embeddings_cache.get(
            key="Hello", collection=embed_model._embeddings_cache_collection()
        )
        is not None
    )

    embd_dict = embeddings_cache.get(
        key="Hello", collection=embed_model._embeddings_cache_collection()
    )
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
        collection=embed_model._embeddings_cache_collection(),
    )
    embed_model.embeddings_cache.put(
        key="Cached2",
        val={"uuid3": [104.0, 101.0, 100.0, 50.0]},
        collection=embed_model._embeddings_cache_collection(),
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
            embd_dict = embeddings_cache.get(
                key=text, collection=embed_model._embeddings_cache_collection()
            )

            first_key = next(iter(embd_dict.keys()))
            assert embd_dict[first_key] == expected_embeddings[i]


@pytest.mark.asyncio
async def test_async_get_with_cache():
    embeddings_cache = SimpleKVStore()
    embed_model = MockEmbedding(embed_dim=4, embeddings_cache=embeddings_cache)
    text = "Hello"

    text_embedding = await embed_model.aget_text_embedding(text)
    assert text_embedding == expected_embedding
    assert (
        embeddings_cache.get(
            key="Hello", collection=embed_model._embeddings_cache_collection()
        )
        is not None
    )

    embd_dict = embeddings_cache.get(
        key="Hello", collection=embed_model._embeddings_cache_collection()
    )
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
        collection=embed_model._embeddings_cache_collection(),
    )
    embed_model.embeddings_cache.put(
        key="Cached2",
        val={"uuid3": [104.0, 101.0, 100.0, 50.0]},
        collection=embed_model._embeddings_cache_collection(),
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
            embd_dict = embeddings_cache.get(
                key=text, collection=embed_model._embeddings_cache_collection()
            )

            first_key = next(iter(embd_dict.keys()))
            assert embd_dict[first_key] == expected_embeddings[i]


def test_cache_is_scoped_to_model():
    """A cache shared by two models must not serve one model's vectors to the other."""
    embeddings_cache = SimpleKVStore()
    model_a = MockEmbedding(
        model_name="model-a", embed_dim=2, embeddings_cache=embeddings_cache
    )
    model_b = MockEmbedding(
        model_name="model-b", embed_dim=3, embeddings_cache=embeddings_cache
    )

    assert model_a.get_text_embedding_batch(["Hello"]) == [[0.5, 0.5]]
    assert model_b.get_text_embedding_batch(["Hello"]) == [[0.5, 0.5, 0.5]]
    assert model_b.get_text_embedding("Hello") == [0.5, 0.5, 0.5]
    assert model_b.get_query_embedding("Hello") == [0.5, 0.5, 0.5]

    # The same model still reuses its entries after a persist/load round trip.
    reloaded = MockEmbedding(
        model_name="model-a",
        embed_dim=2,
        embeddings_cache=SimpleKVStore.from_dict(embeddings_cache.to_dict()),
    )
    with patch.object(reloaded, "_get_text_embeddings") as mock_get_embeddings:
        assert reloaded.get_text_embedding_batch(["Hello"]) == [[0.5, 0.5]]
        mock_get_embeddings.assert_not_called()

    # Model ids such as "BAAI/bge-small-en-v1.5" must still give a collection
    # name that every KV store backend accepts.
    hf_style = MockEmbedding(model_name="BAAI/bge-small-en-v1.5", embed_dim=2)
    assert re.fullmatch(r"[A-Za-z0-9_]+", hf_style._embeddings_cache_collection())


@pytest.mark.asyncio
async def test_async_cache_is_scoped_to_model():
    embeddings_cache = SimpleKVStore()
    model_a = MockEmbedding(
        model_name="model-a", embed_dim=2, embeddings_cache=embeddings_cache
    )
    model_b = MockEmbedding(
        model_name="model-b", embed_dim=3, embeddings_cache=embeddings_cache
    )

    assert await model_a.aget_text_embedding_batch(["Hello"]) == [[0.5, 0.5]]
    assert await model_b.aget_text_embedding_batch(["Hello"]) == [[0.5, 0.5, 0.5]]
    assert await model_b.aget_text_embedding("Hello") == [0.5, 0.5, 0.5]
    assert await model_b.aget_query_embedding("Hello") == [0.5, 0.5, 0.5]


def test_cache_is_scoped_to_class_without_class_name():
    """Subclasses that keep the default class_name and model_name must not collide."""

    class SmallEmbedding(MockEmbedding):
        pass

    class LargeEmbedding(MockEmbedding):
        pass

    embeddings_cache = SimpleKVStore()
    small = SmallEmbedding(embed_dim=2, embeddings_cache=embeddings_cache)
    large = LargeEmbedding(embed_dim=3, embeddings_cache=embeddings_cache)

    assert small.get_text_embedding("Hello") == [0.5, 0.5]
    assert large.get_text_embedding("Hello") == [0.5, 0.5, 0.5]
