import pytest
from llama_index.legacy.llms import ChatMessage
from llama_index.legacy.storage.chat_store.redis_chat_store import RedisChatStore

try:
    from redis import Redis
except ImportError:
    Redis = None  # type: ignore


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_add_messages() -> None:
    """Test adding messages to a chat store."""
    chat_store = RedisChatStore()
    chat_store.delete_messages("user1")
    chat_store.delete_messages("user2")

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))
    chat_store.add_message("user2", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user2", ChatMessage(role="user", content="world"))

    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="world"),
    ]
    assert chat_store.get_messages("user2") == [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="world"),
    ]

    keys = chat_store.get_keys()
    assert "user1" in keys
    assert "user2" in keys

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"), idx=0)
    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="world"),
    ]


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_delete_chat_messages() -> None:
    """Test deleting messages from a chat store."""
    chat_store = RedisChatStore()
    chat_store.delete_messages("user1")
    chat_store.delete_messages("user2")

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))
    chat_store.add_message("user2", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user2", ChatMessage(role="user", content="world"))

    chat_store.delete_messages("user1")

    assert chat_store.get_messages("user1") == []
    assert chat_store.get_messages("user2") == [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="world"),
    ]


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_delete_chat_message() -> None:
    """Test undoing messages from a chat store."""
    chat_store = RedisChatStore()
    chat_store.delete_messages("user1")

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))

    chat_store.delete_last_message("user1")

    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello"),
    ]


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_delete_chat_message_idx() -> None:
    """Test undoing messages from a chat store at a specific idx."""
    chat_store = RedisChatStore()
    chat_store.delete_messages("user1")

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))

    chat_store.delete_message("user1", 0)

    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="world"),
    ]


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_set_messages() -> None:
    chat_store = RedisChatStore()
    chat_store.delete_messages("user1")

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))

    new_messages = [
        ChatMessage(role="user", content="hello2"),
        ChatMessage(role="user", content="world2"),
    ]

    chat_store.set_messages("user1", new_messages)

    new_store = chat_store.get_messages("user1")

    assert len(new_store) == 2
    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello2"),
        ChatMessage(role="user", content="world2"),
    ]
