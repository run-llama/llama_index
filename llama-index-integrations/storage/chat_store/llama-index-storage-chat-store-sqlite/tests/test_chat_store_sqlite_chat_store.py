from pathlib import Path
from typing import Generator

import pytest
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.sqlite import SQLiteChatStore

try:
    import aiosqlite  # noqa
    import sqlalchemy  # noqa
    import sqlite3  # noqa

    no_packages = False
except ImportError:
    no_packages = True


def test_class():
    names_of_base_classes = [b.__name__ for b in SQLiteChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
def sqlite_chat_store(tmp_path: Path) -> Generator[SQLiteChatStore, None, None]:
    chat_store = None
    try:
        chat_store = SQLiteChatStore.from_uri(
            uri=f"sqlite+aiosqlite:///{Path(tmp_path, 'chat_store.db').as_posix()}"
        )
        yield chat_store
    finally:
        if chat_store:
            keys = chat_store.get_keys()
            for key in keys:
                chat_store.delete_messages(key)


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
def test_sqlite_add_message(sqlite_chat_store: SQLiteChatStore):
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    sqlite_chat_store.add_message(key, message=message)

    result = sqlite_chat_store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
def test_set_and_retrieve_messages(sqlite_chat_store: SQLiteChatStore):
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Second message", role="user"),
    ]
    key = "test_set_key"
    sqlite_chat_store.set_messages(key, messages)

    retrieved_messages = sqlite_chat_store.get_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First message"
    assert retrieved_messages[1].content == "Second message"


def test_delete_messages(sqlite_chat_store: SQLiteChatStore):
    messages = [ChatMessage(content="Message to delete", role="user")]
    key = "test_delete_key"
    sqlite_chat_store.set_messages(key, messages)

    sqlite_chat_store.delete_messages(key)
    retrieved_messages = sqlite_chat_store.get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
def test_delete_specific_message(sqlite_chat_store: SQLiteChatStore):
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]
    key = "test_delete_message_key"
    sqlite_chat_store.set_messages(key, messages)

    sqlite_chat_store.delete_message(key, 2)
    retrieved_messages = sqlite_chat_store.get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Keep me"


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
def test_get_keys(sqlite_chat_store: SQLiteChatStore):
    # Add some test data
    sqlite_chat_store.set_messages("key1", [ChatMessage(content="Test1", role="user")])
    sqlite_chat_store.set_messages("key2", [ChatMessage(content="Test2", role="user")])

    keys = sqlite_chat_store.get_keys()
    assert "key1" in keys
    assert "key2" in keys


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
def test_delete_last_message(sqlite_chat_store: SQLiteChatStore):
    key = "test_delete_last_message"
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Last message", role="user"),
    ]
    sqlite_chat_store.set_messages(key, messages)

    deleted_message = sqlite_chat_store.delete_last_message(key)

    assert deleted_message.content == "Last message"

    remaining_messages = sqlite_chat_store.get_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First message"


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_async_sqlite_add_message(sqlite_chat_store: SQLiteChatStore):
    key = "test_async_add_key"

    message = ChatMessage(content="async_add_message_test", role="user")
    await sqlite_chat_store.async_add_message(key, message=message)

    result = await sqlite_chat_store.aget_messages(key)

    assert result[0].content == "async_add_message_test" and result[0].role == "user"


@pytest.mark.asyncio
async def test_async_set_and_retrieve_messages(sqlite_chat_store: SQLiteChatStore):
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Second async message", role="user"),
    ]
    key = "test_async_set_key"
    await sqlite_chat_store.aset_messages(key, messages)

    retrieved_messages = await sqlite_chat_store.aget_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First async message"
    assert retrieved_messages[1].content == "Second async message"


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_adelete_messages(sqlite_chat_store: SQLiteChatStore):
    messages = [ChatMessage(content="Async message to delete", role="user")]
    key = "test_async_delete_key"
    await sqlite_chat_store.aset_messages(key, messages)

    await sqlite_chat_store.adelete_messages(key)
    retrieved_messages = await sqlite_chat_store.aget_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_async_delete_specific_message(sqlite_chat_store: SQLiteChatStore):
    messages = [
        ChatMessage(content="Async keep me", role="user"),
        ChatMessage(content="Async delete me", role="user"),
    ]
    key = "test_adelete_message_key"
    await sqlite_chat_store.aset_messages(key, messages)

    await sqlite_chat_store.adelete_message(key, 2)
    retrieved_messages = await sqlite_chat_store.aget_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Async keep me"


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_async_get_keys(sqlite_chat_store: SQLiteChatStore):
    # Add some test data
    await sqlite_chat_store.aset_messages(
        "async_key1", [ChatMessage(content="Test1", role="user")]
    )
    await sqlite_chat_store.aset_messages(
        "async_key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = await sqlite_chat_store.aget_keys()
    assert "async_key1" in keys
    assert "async_key2" in keys


@pytest.mark.skipif(
    no_packages, reason="aiosqlite, sqlite3 and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_async_delete_last_message(sqlite_chat_store: SQLiteChatStore):
    key = "test_async_delete_last_message"
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Last async message", role="user"),
    ]
    await sqlite_chat_store.aset_messages(key, messages)

    deleted_message = await sqlite_chat_store.adelete_last_message(key)

    assert deleted_message.content == "Last async message"

    remaining_messages = await sqlite_chat_store.aget_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First async message"
