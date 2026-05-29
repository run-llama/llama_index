"""Tests for CockroachDBDocumentStore."""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.schema import TextNode

from llama_index.storage.docstore.cockroachdb import CockroachDBDocumentStore


@pytest.fixture()
def docstore(fresh_db: dict[str, Any]) -> CockroachDBDocumentStore:
    return CockroachDBDocumentStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="docs",
    )


def test_add_get_delete(docstore: CockroachDBDocumentStore) -> None:
    nodes = [
        TextNode(text="hello", id_="a"),
        TextNode(text="world", id_="b"),
    ]
    docstore.add_documents(nodes)

    fetched = docstore.get_document("a")
    assert fetched is not None
    assert fetched.get_content() == "hello"

    docstore.delete_document("a")
    assert docstore.get_document("a", raise_error=False) is None
    assert docstore.get_document("b") is not None


@pytest.mark.asyncio
async def test_async_add_get(fresh_db: dict[str, Any]) -> None:
    store = CockroachDBDocumentStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="docs_async",
    )
    await store.async_add_documents([TextNode(text="alpha", id_="x")])
    fetched = await store.aget_document("x")
    assert fetched is not None
    assert fetched.get_content() == "alpha"
