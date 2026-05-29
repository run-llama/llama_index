"""Tests for CockroachDBKVStore (sync + async)."""

from __future__ import annotations

from typing import Any

import pytest

from llama_index.storage.kvstore.cockroachdb import CockroachDBKVStore


@pytest.fixture()
def kvstore(fresh_db: dict[str, Any]) -> CockroachDBKVStore:
    return CockroachDBKVStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="kv_test",
    )


def test_put_get_delete(kvstore: CockroachDBKVStore) -> None:
    kvstore.put("alpha", {"n": 1})
    kvstore.put("beta", {"n": 2})

    assert kvstore.get("alpha") == {"n": 1}
    assert kvstore.get("missing") is None

    all_rows = kvstore.get_all()
    assert all_rows == {"alpha": {"n": 1}, "beta": {"n": 2}}

    assert kvstore.delete("alpha") is True
    assert kvstore.delete("alpha") is False
    assert kvstore.get("alpha") is None


def test_collection_isolation(kvstore: CockroachDBKVStore) -> None:
    kvstore.put("k", {"v": 1}, collection="a")
    kvstore.put("k", {"v": 2}, collection="b")
    assert kvstore.get("k", collection="a") == {"v": 1}
    assert kvstore.get("k", collection="b") == {"v": 2}


def test_put_all_upsert(kvstore: CockroachDBKVStore) -> None:
    kvstore.put_all([("k1", {"x": 1}), ("k2", {"x": 2})])
    kvstore.put_all([("k1", {"x": 10})])
    assert kvstore.get("k1") == {"x": 10}
    assert kvstore.get("k2") == {"x": 2}


@pytest.mark.asyncio
async def test_async_round_trip(fresh_db: dict[str, Any]) -> None:
    store = CockroachDBKVStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="kv_async",
    )
    await store.aput("alpha", {"n": 1})
    await store.aput_all([("beta", {"n": 2}), ("gamma", {"n": 3})])

    assert await store.aget("alpha") == {"n": 1}
    assert await store.aget("missing") is None

    rows = await store.aget_all()
    assert rows == {"alpha": {"n": 1}, "beta": {"n": 2}, "gamma": {"n": 3}}

    assert await store.adelete("beta") is True
    assert await store.adelete("beta") is False
