from pathlib import Path

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore


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


def test_delete_chat_message_out_of_bounds() -> None:
    """Test deleting a message with an out-of-bounds negative index."""
    chat_store = SimpleChatStore()
    chat_store.add_message("user1", ChatMessage(role="user", content="hello"))

    assert chat_store.delete_message("user1", -2) is None
    assert chat_store.get_messages("user1") == [
        ChatMessage(role="user", content="hello"),
    ]


def test_delete_last_chat_message_from_empty_history() -> None:
    """Test deleting the last message from an existing empty history."""
    chat_store = SimpleChatStore()
    message = ChatMessage(role="user", content="hello")
    chat_store.add_message("user1", message)

    assert chat_store.delete_last_message("user1") == message
    assert chat_store.delete_last_message("user1") is None


def test_persist_non_ascii_unescaped(tmp_path: Path) -> None:
    """Test that non-ascii content is persisted as-is, not as unicode escapes."""
    chat_store = SimpleChatStore()

    chat_store.add_message("user1", ChatMessage(role="user", content="ハロー héllo"))

    persist_path = str(tmp_path / "chat_store.json")
    chat_store.persist(persist_path)

    assert "ハロー héllo" in (tmp_path / "chat_store.json").read_text(encoding="utf-8")
    assert SimpleChatStore.from_persist_path(persist_path).get_messages("user1") == [
        ChatMessage(role="user", content="ハロー héllo"),
    ]
