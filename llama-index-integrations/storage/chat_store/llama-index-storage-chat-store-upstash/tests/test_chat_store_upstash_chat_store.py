import pytest
from llama_index.storage.chat_store.upstash import UpstashChatStore
import os
from importlib.util import find_spec
from llama_index.core.llms import ChatMessage
import time
import asyncio
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def dict_to_message(d: dict) -> ChatMessage:
    return ChatMessage.parse_obj(d)


try:
    find_spec("upstash-redis")
    if os.environ.get("UPSTASH_REDIS_REST_URL") and os.environ.get(
        "UPSTASH_REDIS_REST_TOKEN"
    ):
        upstash_installed = True
    else:
        upstash_installed = False
except ImportError:
    upstash_installed = False


@pytest.fixture()
def upstash_chat_store() -> UpstashChatStore:
    return UpstashChatStore(
        redis_url=os.environ.get("UPSTASH_REDIS_REST_URL") or "",
        redis_token=os.environ.get("UPSTASH_REDIS_REST_TOKEN") or "",
    )


####################
#### SYNC TESTS ####
####################
@pytest.mark.skip(reason="Skipping all tests")
def test_invalid_initialization():
    with pytest.raises(ValueError):
        UpstashChatStore(redis_url="", redis_token="")


@pytest.mark.skip(reason="Skipping all tests")
def test_upstash_basic(upstash_chat_store: UpstashChatStore):
    assert upstash_chat_store.class_name() == "UpstashChatStore"


@pytest.mark.skip(reason="Skipping all tests")
def test_upstash_add_message(upstash_chat_store: UpstashChatStore):
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    upstash_chat_store.add_message(key, message=message)

    result = upstash_chat_store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


@pytest.mark.skip(reason="Skipping all tests")
def test_set_and_retrieve_messages(upstash_chat_store: UpstashChatStore):
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Second message", role="user"),
    ]
    key = "test_set_key"
    upstash_chat_store.set_messages(key, messages)

    retrieved_messages = upstash_chat_store.get_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First message"
    assert retrieved_messages[1].content == "Second message"


@pytest.mark.skip(reason="Skipping all tests")
def test_delete_messages(upstash_chat_store: UpstashChatStore):
    messages = [ChatMessage(content="Message to delete", role="user")]
    key = "test_delete_key"
    upstash_chat_store.set_messages(key, messages)

    upstash_chat_store.delete_messages(key)
    retrieved_messages = upstash_chat_store.get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skip(reason="Skipping all tests")
def test_delete_specific_message(upstash_chat_store: UpstashChatStore):
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]
    key = "test_delete_message_key"
    upstash_chat_store.set_messages(key, messages)

    upstash_chat_store.delete_message(key, 1)
    retrieved_messages = upstash_chat_store.get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Keep me"


@pytest.mark.skip(reason="Skipping all tests")
def test_ttl_on_messages(upstash_chat_store: UpstashChatStore):
    upstash_chat_store.ttl = 3
    key = "ttl_test_key"
    message = ChatMessage(content="This message will expire", role="user")
    upstash_chat_store.add_message(key, message)

    time.sleep(4)  # Waiting for the ttl to expire.

    retrieved_messages = upstash_chat_store.get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skip(reason="Skipping all tests")
def test_add_message_at_index(upstash_chat_store: UpstashChatStore):
    key = "test_add_message_index_key"
    # Clear any existing data for the key
    upstash_chat_store.delete_messages(key)

    # Initial messages to add
    initial_messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Third message", role="user"),
    ]

    upstash_chat_store.set_messages(key, initial_messages)

    # Message to insert at index 1
    new_message = ChatMessage(content="Second message", role="user")
    upstash_chat_store.add_message(key, new_message, idx=1)

    # Retrieve messages to check the order
    result_messages = upstash_chat_store.get_messages(key)
    assert len(result_messages) == 3
    assert result_messages[0].content == "First message"
    assert result_messages[1].content == "Second message"
    assert result_messages[2].content == "Third message"


#####################
#### ASYNC TESTS ####
#####################


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_upstash_add_message(upstash_chat_store: UpstashChatStore):
    key = "test_async_add_key"

    message = ChatMessage(content="async_add_message_test", role="user")
    await upstash_chat_store.async_add_message(key, message=message)

    result = await upstash_chat_store.async_get_messages(key)

    assert result[0].content == "async_add_message_test" and result[0].role == "user"


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_set_and_retrieve_messages(upstash_chat_store: UpstashChatStore):
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Second async message", role="user"),
    ]
    key = "test_async_set_key"
    await upstash_chat_store.async_set_messages(key, messages)

    retrieved_messages = await upstash_chat_store.async_get_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First async message"
    assert retrieved_messages[1].content == "Second async message"


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_delete_messages(upstash_chat_store: UpstashChatStore):
    messages = [ChatMessage(content="Async message to delete", role="user")]
    key = "test_async_delete_key"
    await upstash_chat_store.async_set_messages(key, messages)

    await upstash_chat_store.async_delete_messages(key)
    retrieved_messages = await upstash_chat_store.async_get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_delete_specific_message(upstash_chat_store: UpstashChatStore):
    messages = [
        ChatMessage(content="Async keep me", role="user"),
        ChatMessage(content="Async delete me", role="user"),
    ]
    key = "test_async_delete_message_key"
    await upstash_chat_store.async_set_messages(key, messages)

    await upstash_chat_store.async_delete_message(key, 1)
    retrieved_messages = await upstash_chat_store.async_get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Async keep me"


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_ttl_on_messages(upstash_chat_store: UpstashChatStore):
    upstash_chat_store.ttl = 3
    key = "async_ttl_test_key"
    message = ChatMessage(content="This async message will expire", role="user")
    await upstash_chat_store.async_add_message(key, message)

    await asyncio.sleep(4)  # Waiting for the ttl to expire.

    retrieved_messages = await upstash_chat_store.async_get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_add_message_at_index(upstash_chat_store: UpstashChatStore):
    key = "test_async_add_message_index_key"
    # Clear any existing data for the key
    await upstash_chat_store.async_delete_messages(key)

    initial_messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Third async message", role="user"),
    ]

    await upstash_chat_store.async_set_messages(key, initial_messages)

    new_message = ChatMessage(content="Second async message", role="user")
    await upstash_chat_store.async_add_message(key, new_message, idx=1)

    # Retrieve messages to check the order
    result_messages = await upstash_chat_store.async_get_messages(key)
    assert len(result_messages) == 3
    assert result_messages[0].content == "First async message"
    assert result_messages[1].content == "Second async message"
    assert result_messages[2].content == "Third async message"


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_get_keys(upstash_chat_store: UpstashChatStore):
    # Add some test data
    await upstash_chat_store.async_set_messages(
        "async_key1", [ChatMessage(content="Test1", role="user")]
    )
    await upstash_chat_store.async_set_messages(
        "async_key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = await upstash_chat_store.async_get_keys()
    assert "async_key1" in keys
    assert "async_key2" in keys


@pytest.mark.skip(reason="Skipping all tests")
@pytest.mark.asyncio
async def test_async_delete_last_message(upstash_chat_store: UpstashChatStore):
    key = "test_async_delete_last_message"
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Last async message", role="user"),
    ]
    await upstash_chat_store.async_set_messages(key, messages)

    deleted_message = await upstash_chat_store.async_delete_last_message(key)

    assert deleted_message.content == "Last async message"

    remaining_messages = await upstash_chat_store.async_get_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First async message"
