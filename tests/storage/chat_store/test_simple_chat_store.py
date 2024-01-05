from llama_index.llms import ChatMessage
from llama_index.storage.chat_store import SimpleChatStore


def test_add_messages() -> None:
    """Test adding messages to a chat store."""
    chat_store = SimpleChatStore()

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

    assert chat_store.get_keys() == ["user1", "user2"]

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"), idx=0)
    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="user", content="world"),
    ]


def test_delete_chat_messages() -> None:
    """Test deleting messages from a chat store."""
    chat_store = SimpleChatStore()

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


def test_delete_chat_message() -> None:
    """Test undoing messages from a chat store."""
    chat_store = SimpleChatStore()

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))

    chat_store.delete_last_message("user1")

    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello"),
    ]


def test_delete_chat_message_idx() -> None:
    """Test undoing messages from a chat store at a specific idx."""
    chat_store = SimpleChatStore()

    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    chat_store.add_message("user1", ChatMessage(role="user", content="world"))

    chat_store.delete_message("user1", 0)

    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="world"),
    ]
