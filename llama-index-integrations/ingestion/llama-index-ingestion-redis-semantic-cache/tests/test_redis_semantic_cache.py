import pytest
import logging

from llama_index.core.embeddings import MockEmbedding
from llama_index.ingestion.redis_semantic_cache import (
    LlamaIndexVectorizer,
    RedisSemanticCache,
)

logging.basicConfig(level=logging.INFO)


EMBEDDING_DIMS = 5
MOCK_ENTRIES = [
    {
        "query": "Hi! how are you?",
        "response": "I'm doing well, thanks for asking!",
        "metadata": {
            "source": "greeting_conversation",
        },
    }
]
MOCK_EMBED_MODEL = MockEmbedding(embed_dim=EMBEDDING_DIMS)
REDIS_URL = "redis://localhost:6379/0"


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_cache(**kwargs) -> RedisSemanticCache:
    """Create a RedisSemanticCache with default test settings."""
    defaults = {
        "embed_model": MOCK_EMBED_MODEL,
        "embedding_dims": EMBEDDING_DIMS,
        "redis_url": REDIS_URL,
        **kwargs,
    }
    return RedisSemanticCache(**defaults)


def _store_default_entry(cache: RedisSemanticCache) -> None:
    """Store the first MOCK_ENTRIES item into *cache*."""
    cache.store_cache_entry(
        query=MOCK_ENTRIES[0]["query"],
        response=MOCK_ENTRIES[0]["response"],
        metadata=MOCK_ENTRIES[0]["metadata"],
    )


async def _astore_default_entry(cache: RedisSemanticCache) -> None:
    """Async-store the first MOCK_ENTRIES item into *cache*."""
    await cache.astore_cache_entry(
        query=MOCK_ENTRIES[0]["query"],
        response=MOCK_ENTRIES[0]["response"],
        metadata=MOCK_ENTRIES[0]["metadata"],
    )


# ══════════════════════════════════════════════════════════════════════════
# Sync – Vectorizer
# ══════════════════════════════════════════════════════════════════════════


def test_vectorizer_encode():
    """encode() output matches the raw embed model."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    text = "hello world!"
    assert vectorizer.encode(text) == MOCK_EMBED_MODEL.get_text_embedding(text)


def test_vectorizer_embed():
    """embed() (with _process_embedding) still matches the raw embed model."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    text = "hello world!"
    assert vectorizer.embed(text) == MOCK_EMBED_MODEL.get_text_embedding(text)


def test_vectorizer_embed_many():
    """embed_many() returns one embedding per input text."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    texts = ["hello world!", "how are you?"]
    result = vectorizer.embed_many(texts)
    assert len(result) == len(texts)


def test_vectorizer_embed_type_error():
    """embed() raises TypeError on non-str input."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    with pytest.raises(TypeError, match="embed\\(\\) requires a str value"):
        vectorizer.embed(123)


def test_vectorizer_embed_many_type_error():
    """embed_many() raises TypeError on non-list input."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    with pytest.raises(TypeError, match="embed_many\\(\\) requires a list"):
        vectorizer.embed_many("not a list")


# ══════════════════════════════════════════════════════════════════════════
# Sync – RedisSemanticCache
# ══════════════════════════════════════════════════════════════════════════


def test_store_entry():
    cache = _make_cache()
    _store_default_entry(cache)
    assert len(cache) == 1
    cache.delete_cache()


def test_retrieve_entry():
    cache = _make_cache()
    _store_default_entry(cache)
    keys = cache.get_all_cache_keys()
    entry = cache.retrieve_cache_entry(keys[0])

    assert entry["prompt"] == MOCK_ENTRIES[0]["query"]
    assert entry["response"] == MOCK_ENTRIES[0]["response"]
    assert entry["metadata"] == MOCK_ENTRIES[0]["metadata"]
    cache.delete_cache()


def test_check_match():
    cache = _make_cache()
    _store_default_entry(cache)
    results = cache.check(query="Hello, how are you doing?", num_results=1)
    assert len(results.matches) == 1
    cache.delete_cache()


def test_check_no_match():
    """check() returns empty matches when the cache is empty."""
    cache = _make_cache()
    results = cache.check(query="Anything at all", num_results=1)
    assert len(results.matches) == 0
    cache.delete_cache()


def test_check_with_distance_threshold():
    """check() respects a custom distance_threshold."""
    cache = _make_cache()
    _store_default_entry(cache)
    results = cache.check(
        query="Hello, how are you doing?",
        distance_threshold=1.0,
        num_results=1,
    )
    assert len(results.matches) >= 1
    cache.delete_cache()


def test_update_entry():
    cache = _make_cache()
    _store_default_entry(cache)
    keys = cache.get_all_cache_keys()
    assert len(keys) > 0

    cache.update(key=keys[0], metadata={"hit_count": 5})
    entry = cache.retrieve_cache_entry(keys[0])
    assert entry["metadata"]["hit_count"] == 5
    cache.delete_cache()


def test_store_with_custom_ttl():
    """store_cache_entry() accepts a custom TTL override."""
    cache = _make_cache()
    cache.store_cache_entry(
        query=MOCK_ENTRIES[0]["query"],
        response=MOCK_ENTRIES[0]["response"],
        ttl=60,
    )
    assert len(cache) == 1
    cache.delete_cache()


def test_remove_entries_by_key():
    cache = _make_cache()
    _store_default_entry(cache)
    keys = cache.get_all_cache_keys()
    cache.remove_cache_entries(keys=keys)
    assert len(cache) == 0
    cache.delete_cache()


def test_remove_entries_raises_without_args():
    """remove_cache_entries() raises ValueError when neither ids nor keys provided."""
    cache = _make_cache()
    with pytest.raises(
        ValueError, match="At least one of ids or keys must be provided"
    ):
        cache.remove_cache_entries()
    cache.delete_cache()


def test_clear_and_delete():
    cache = _make_cache()
    _store_default_entry(cache)
    assert len(cache) == 1

    cache.clear_cache()
    assert len(cache) == 0

    cache.delete_cache()


# ══════════════════════════════════════════════════════════════════════════
# Async – Vectorizer
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_vectorizer_aencode():
    """aencode() output matches aget_text_embedding."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    text = "hello world!"
    embed_async = await vectorizer.aencode(text)
    embed_sync = await MOCK_EMBED_MODEL.aget_text_embedding(text)
    assert embed_async == embed_sync


@pytest.mark.asyncio
async def test_vectorizer_aembed():
    """aembed() matches embed() for the same input."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    text = "hello world!"
    embed_async = await vectorizer.aembed(text)
    embed_sync = vectorizer.embed(text)
    assert embed_async == embed_sync


@pytest.mark.asyncio
async def test_vectorizer_aembed_many():
    """aembed_many() matches embed_many() for the same inputs."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    texts = ["hello world!", "how are you?"]
    embeds_async = await vectorizer.aembed_many(texts)
    embeds_sync = vectorizer.embed_many(texts)
    assert embeds_async == embeds_sync


@pytest.mark.asyncio
async def test_vectorizer_aembed_type_error():
    """aembed() raises TypeError on non-str input."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    with pytest.raises(TypeError, match="aembed\\(\\) requires a str value"):
        await vectorizer.aembed(123)


@pytest.mark.asyncio
async def test_vectorizer_aembed_many_type_error():
    """aembed_many() raises TypeError on non-list input."""
    vectorizer = LlamaIndexVectorizer(
        embed_model=MOCK_EMBED_MODEL,
        embedding_dims=EMBEDDING_DIMS,
    )
    with pytest.raises(TypeError, match="aembed_many\\(\\) requires a list"):
        await vectorizer.aembed_many("not a list")


# ══════════════════════════════════════════════════════════════════════════
# Async – RedisSemanticCache
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_async_store_cache_entry():
    cache = _make_cache()
    await _astore_default_entry(cache)
    assert len(cache) == 1
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_check():
    cache = _make_cache()
    await _astore_default_entry(cache)
    results = await cache.acheck(query="Hello, how are you doing?", num_results=1)
    assert len(results.matches) == 1
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_check_no_match():
    """acheck() returns empty matches when the cache is empty."""
    cache = _make_cache()
    results = await cache.acheck(query="Anything at all", num_results=1)
    assert len(results.matches) == 0
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_update():
    cache = _make_cache()
    await _astore_default_entry(cache)
    keys = cache.get_all_cache_keys()
    assert len(keys) > 0, "Expected at least one cache key to update"

    await cache.aupdate(
        key=keys[0],
        metadata={"hit_count": 1, "model_name": "test-model"},
    )
    entry = cache.retrieve_cache_entry(keys[0])
    assert entry["metadata"]["hit_count"] == 1
    assert entry["metadata"]["model_name"] == "test-model"
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_store_with_custom_ttl():
    """astore_cache_entry() accepts a custom TTL override."""
    cache = _make_cache()
    await cache.astore_cache_entry(
        query=MOCK_ENTRIES[0]["query"],
        response=MOCK_ENTRIES[0]["response"],
        ttl=60,
    )
    assert len(cache) == 1
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_remove_cache_entries():
    cache = _make_cache()
    await _astore_default_entry(cache)
    keys = cache.get_all_cache_keys()
    await cache.aremove_cache_entries(keys=keys)
    assert len(cache) == 0
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_remove_cache_entries_raises():
    cache = _make_cache()
    with pytest.raises(
        ValueError, match="At least one of ids or keys must be provided"
    ):
        await cache.aremove_cache_entries()
    cache.delete_cache()


@pytest.mark.asyncio
async def test_async_clear_and_delete():
    cache = _make_cache()
    await _astore_default_entry(cache)
    assert len(cache) == 1

    await cache.aclear_cache()
    assert len(cache) == 0

    await cache.adelete_cache()
