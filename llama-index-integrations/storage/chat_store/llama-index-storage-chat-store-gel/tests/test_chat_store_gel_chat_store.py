import subprocess
import pytest
import pytest_asyncio
import os
from typing import Generator
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.gel import GelChatStore

skip_in_cicd = os.environ.get("CI") is not None

try:
    if not skip_in_cicd:
        subprocess.run(["gel", "project", "init", "--non-interactive"], check=True)
except subprocess.CalledProcessError as e:
    print(e)


def test_class():
    names_of_base_classes = [b.__name__ for b in GelChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
def gel_chat_store() -> Generator[GelChatStore, None, None]:
    chat_store = None
    try:
        chat_store = GelChatStore()
        yield chat_store
    finally:
        if chat_store:
            keys = chat_store.get_keys()
            for key in keys:
                chat_store.delete_messages(key)


@pytest_asyncio.fixture()
async def gel_chat_store_async():
    # New instance of the GelKVStore client to use it in async mode
    chat_store = None
    try:
        chat_store = GelChatStore()
        yield chat_store
    finally:
        if chat_store:
            keys = await chat_store.aget_keys()
            for key in keys:
                await chat_store.adelete_messages(key)


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
def test_gel_add_message(gel_chat_store: GelChatStore):
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    gel_chat_store.add_message(key, message=message)

    result = gel_chat_store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
def test_set_and_retrieve_messages(gel_chat_store: GelChatStore):
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Second message", role="user"),
    ]
    key = "test_set_key"
    gel_chat_store.set_messages(key, messages)

    retrieved_messages = gel_chat_store.get_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First message"
    assert retrieved_messages[1].content == "Second message"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
def test_delete_messages(gel_chat_store: GelChatStore):
    messages = [ChatMessage(content="Message to delete", role="user")]
    key = "test_delete_key"
    gel_chat_store.set_messages(key, messages)

    gel_chat_store.delete_messages(key)
    retrieved_messages = gel_chat_store.get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
def test_delete_specific_message(gel_chat_store: GelChatStore):
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]
    key = "test_delete_message_key"
    gel_chat_store.set_messages(key, messages)

    deleted_message = gel_chat_store.delete_message(key, 1)
    retrieved_messages = gel_chat_store.get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Keep me"
    assert deleted_message.content == "Delete me"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
def test_get_keys(gel_chat_store: GelChatStore):
    # Add some test data
    gel_chat_store.set_messages("key1", [ChatMessage(content="Test1", role="user")])
    gel_chat_store.set_messages("key2", [ChatMessage(content="Test2", role="user")])

    keys = gel_chat_store.get_keys()
    assert "key1" in keys
    assert "key2" in keys


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
def test_delete_last_message(gel_chat_store: GelChatStore):
    key = "test_delete_last_message"
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Last message", role="user"),
    ]
    gel_chat_store.set_messages(key, messages)

    deleted_message = gel_chat_store.delete_last_message(key)

    assert deleted_message.content == "Last message"

    remaining_messages = gel_chat_store.get_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First message"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
@pytest.mark.asyncio
async def test_async_gel_add_message(gel_chat_store_async: GelChatStore):
    key = "test_async_add_key"

    message = ChatMessage(content="async_add_message_test", role="user")
    await gel_chat_store_async.async_add_message(key, message=message)

    result = await gel_chat_store_async.aget_messages(key)

    assert result[0].content == "async_add_message_test" and result[0].role == "user"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
@pytest.mark.asyncio
async def test_async_set_and_retrieve_messages(gel_chat_store_async: GelChatStore):
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Second async message", role="user"),
    ]
    key = "test_async_set_key"
    await gel_chat_store_async.aset_messages(key, messages)

    retrieved_messages = await gel_chat_store_async.aget_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First async message"
    assert retrieved_messages[1].content == "Second async message"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
@pytest.mark.asyncio
async def test_async_delete_messages(gel_chat_store_async: GelChatStore):
    messages = [ChatMessage(content="Async message to delete", role="user")]
    key = "test_async_delete_key"
    await gel_chat_store_async.aset_messages(key, messages)

    await gel_chat_store_async.adelete_messages(key)
    retrieved_messages = await gel_chat_store_async.aget_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
@pytest.mark.asyncio
async def test_async_delete_specific_message(gel_chat_store_async: GelChatStore):
    messages = [
        ChatMessage(content="Async keep me", role="user"),
        ChatMessage(content="Async delete me", role="user"),
    ]
    key = "test_adelete_message_key"
    await gel_chat_store_async.aset_messages(key, messages)

    deleted_message = await gel_chat_store_async.adelete_message(key, 1)
    retrieved_messages = await gel_chat_store_async.aget_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Async keep me"
    assert deleted_message.content == "Async delete me"


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
@pytest.mark.asyncio
async def test_async_get_keys(gel_chat_store_async: GelChatStore):
    # Add some test data
    await gel_chat_store_async.aset_messages(
        "async_key1", [ChatMessage(content="Test1", role="user")]
    )
    await gel_chat_store_async.aset_messages(
        "async_key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = await gel_chat_store_async.aget_keys()
    assert "async_key1" in keys
    assert "async_key2" in keys


@pytest.mark.skipif(skip_in_cicd, reason="gel package not installed")
@pytest.mark.asyncio
async def test_async_delete_last_message(gel_chat_store_async: GelChatStore):
    key = "test_async_delete_last_message"
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Last async message", role="user"),
    ]
    await gel_chat_store_async.aset_messages(key, messages)

    deleted_message = await gel_chat_store_async.adelete_last_message(key)

    assert deleted_message.content == "Last async message"

    remaining_messages = await gel_chat_store_async.aget_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First async message"
