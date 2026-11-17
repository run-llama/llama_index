"""Tests for deriving the async client from a sync client or URL."""

import pytest
from redis import Redis
from redis.asyncio import Redis as RedisAsync
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ConstantBackoff
from redis.retry import Retry as SyncRetry
from redis.sentinel import Sentinel

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.base import _async_client_from_sync


def _async_kwargs(sync_client: Redis) -> dict:
    return _async_client_from_sync(sync_client).connection_pool.connection_kwargs


def test_url_settings_survive():
    kwargs = _async_kwargs(Redis.from_url("redis://user:secret@db.example:6380/2"))
    assert kwargs["host"] == "db.example"
    assert kwargs["port"] == 6380
    assert kwargs["db"] == 2
    assert kwargs["username"] == "user"
    assert kwargs["password"] == "secret"


def test_tls_url_keeps_tls():
    client = _async_client_from_sync(
        Redis.from_url("rediss://db.example:6380/0?ssl_cert_reqs=required")
    )
    pool = client.connection_pool
    assert pool.connection_class.__name__ == "SSLConnection"
    assert pool.connection_kwargs["ssl_cert_reqs"] == "required"


def test_tls_constructor_keeps_tls():
    client = _async_client_from_sync(
        Redis(host="db.example", ssl=True, ssl_ca_certs="/etc/ca.pem")
    )
    pool = client.connection_pool
    assert pool.connection_class.__name__ == "SSLConnection"
    assert pool.connection_kwargs["ssl_ca_certs"] == "/etc/ca.pem"


def test_unix_socket():
    client = _async_client_from_sync(Redis.from_url("unix:///tmp/redis.sock?db=1"))
    pool = client.connection_pool
    assert pool.connection_class.__name__ == "UnixDomainSocketConnection"
    assert pool.connection_kwargs["path"] == "/tmp/redis.sock"
    assert pool.connection_kwargs["db"] == 1


def test_resp3():
    assert _async_kwargs(Redis(host="db.example", protocol=3))["protocol"] == 3


def test_sync_retry_becomes_async_retry():
    sync_client = Redis(host="db.example", retry=SyncRetry(ConstantBackoff(0.5), 7))
    retry = _async_kwargs(sync_client)["retry"]
    assert isinstance(retry, AsyncRetry)
    assert retry._retries == 7
    assert isinstance(retry._backoff, ConstantBackoff)


def test_unsupported_setting_raises():
    with pytest.raises(ValueError, match="ssl_validate_ocsp"):
        _async_client_from_sync(
            Redis(host="db.example", ssl=True, ssl_validate_ocsp=True)
        )


def test_sentinel_client_raises():
    sync_client = Sentinel([("db.example", 26379)]).master_for("mymaster")
    with pytest.raises(ValueError, match="redis_client_async"):
        _async_client_from_sync(sync_client)


def test_client_without_pool_raises():
    with pytest.raises(ValueError, match="redis_client_async"):
        _async_client_from_sync(object())


def test_url_only_store_builds_async_client_from_url():
    vector_store = RedisVectorStore(redis_url="redis://localhost:6379/0")
    assert isinstance(vector_store._redis_client_async, RedisAsync)


@pytest.mark.asyncio
async def test_async_round_trip_with_sync_client_only(documents, turtle_test):
    vector_store = RedisVectorStore(
        redis_client=Redis.from_url("redis://localhost:6379/0"), overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )
    for doc in documents:
        await index.ainsert(doc)
    retriever = index.as_retriever(similarity_top_k=1)
    result_nodes = await retriever.aretrieve(turtle_test["question"])
    assert result_nodes[0].metadata == turtle_test["metadata"]

    for doc in documents:
        await vector_store.adelete(doc.doc_id)
