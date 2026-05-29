"""Tests for CockroachDBIndexStore."""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.data_structs.data_structs import IndexDict

from llama_index.storage.index_store.cockroachdb import CockroachDBIndexStore


@pytest.fixture()
def index_store(fresh_db: dict[str, Any]) -> CockroachDBIndexStore:
    return CockroachDBIndexStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="idx",
    )


def test_add_get_delete(index_store: CockroachDBIndexStore) -> None:
    struct = IndexDict(index_id="my-index")
    index_store.add_index_struct(struct)

    fetched = index_store.get_index_struct("my-index")
    assert fetched is not None
    assert fetched.index_id == "my-index"

    structs = index_store.index_structs()
    assert any(s.index_id == "my-index" for s in structs)

    index_store.delete_index_struct("my-index")
    assert index_store.get_index_struct("my-index") is None


@pytest.mark.asyncio
async def test_async_add_get(fresh_db: dict[str, Any]) -> None:
    store = CockroachDBIndexStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="idx_async",
    )
    struct = IndexDict(index_id="async-idx")
    await store.async_add_index_struct(struct)
    fetched = await store.aget_index_struct("async-idx")
    assert fetched is not None
    assert fetched.index_id == "async-idx"
