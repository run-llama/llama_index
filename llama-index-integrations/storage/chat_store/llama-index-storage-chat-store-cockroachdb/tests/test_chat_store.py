"""Tests for CockroachDBChatStore."""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from llama_index.storage.chat_store.cockroachdb import CockroachDBChatStore


@pytest.fixture()
def chat_store(fresh_db: dict[str, Any]) -> CockroachDBChatStore:
    return CockroachDBChatStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="chats",
    )


def _msg(role: MessageRole, text: str) -> ChatMessage:
    return ChatMessage(role=role, content=text)


def test_set_and_get_messages(chat_store: CockroachDBChatStore) -> None:
    msgs = [_msg(MessageRole.USER, "hi"), _msg(MessageRole.ASSISTANT, "hello")]
    chat_store.set_messages("sess-1", msgs)
    out = chat_store.get_messages("sess-1")
    assert [m.content for m in out] == ["hi", "hello"]
    assert [m.role for m in out] == [MessageRole.USER, MessageRole.ASSISTANT]


def test_add_message_appends(chat_store: CockroachDBChatStore) -> None:
    chat_store.add_message("sess-2", _msg(MessageRole.USER, "first"))
    chat_store.add_message("sess-2", _msg(MessageRole.ASSISTANT, "second"))
    chat_store.add_message("sess-2", _msg(MessageRole.USER, "third"))
    out = chat_store.get_messages("sess-2")
    assert [m.content for m in out] == ["first", "second", "third"]


def test_delete_message_by_index(chat_store: CockroachDBChatStore) -> None:
    chat_store.set_messages(
        "sess-3",
        [
            _msg(MessageRole.USER, "a"),
            _msg(MessageRole.USER, "b"),
            _msg(MessageRole.USER, "c"),
        ],
    )
    removed = chat_store.delete_message("sess-3", 1)
    assert removed is not None
    assert removed.content == "b"
    out = chat_store.get_messages("sess-3")
    assert [m.content for m in out] == ["a", "c"]

    assert chat_store.delete_message("sess-3", 99) is None


def test_delete_last_message(chat_store: CockroachDBChatStore) -> None:
    chat_store.set_messages(
        "sess-4",
        [_msg(MessageRole.USER, "x"), _msg(MessageRole.USER, "y")],
    )
    removed = chat_store.delete_last_message("sess-4")
    assert removed is not None
    assert removed.content == "y"
    out = chat_store.get_messages("sess-4")
    assert [m.content for m in out] == ["x"]


def test_delete_all_and_keys(chat_store: CockroachDBChatStore) -> None:
    chat_store.set_messages("k1", [_msg(MessageRole.USER, "a")])
    chat_store.set_messages("k2", [_msg(MessageRole.USER, "b")])
    keys = set(chat_store.get_keys())
    assert {"k1", "k2"}.issubset(keys)

    chat_store.delete_messages("k1")
    assert chat_store.get_messages("k1") == []


@pytest.mark.asyncio
async def test_async_round_trip(fresh_db: dict[str, Any]) -> None:
    store = CockroachDBChatStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        database=fresh_db["database"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        sslmode="disable",
        table_name="chats_async",
    )
    await store.aset_messages(
        "s1",
        [_msg(MessageRole.USER, "hi"), _msg(MessageRole.ASSISTANT, "hey")],
    )
    out = await store.aget_messages("s1")
    assert [m.content for m in out] == ["hi", "hey"]

    await store.async_add_message("s1", _msg(MessageRole.USER, "again"))
    out = await store.aget_messages("s1")
    assert [m.content for m in out] == ["hi", "hey", "again"]

    removed = await store.adelete_last_message("s1")
    assert removed is not None
    assert removed.content == "again"

    await store.adelete_messages("s1")
    assert await store.aget_messages("s1") == []
