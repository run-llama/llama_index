import pathlib

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


def test_persist_uses_utf8_and_no_ascii_escaping(tmp_path: pathlib.Path) -> None:
    chat_store = SimpleChatStore()
    chat_store.add_message("k", ChatMessage(role="user", content="café 你好"))

    persist_path = str(tmp_path / "chat_store.json")
    chat_store.persist(persist_path=persist_path)

    raw = (tmp_path / "chat_store.json").read_text(encoding="utf-8")
    assert "café 你好" in raw
    assert "\\u" not in raw

    loaded = SimpleChatStore.from_persist_path(persist_path=persist_path)
    assert loaded.get_messages("k")[0].content == "café 你好"


def test_persist_and_load_with_custom_encoding(tmp_path: pathlib.Path) -> None:
    chat_store = SimpleChatStore()
    chat_store.add_message("k", ChatMessage(role="user", content="Olá こんにちは"))

    persist_path = str(tmp_path / "nested" / "chat_store.json")
    chat_store.persist(persist_path=persist_path, encoding="utf-16")

    loaded = SimpleChatStore.from_persist_path(
        persist_path=persist_path, encoding="utf-16"
    )
    assert loaded.get_messages("k")[0].content == "Olá こんにちは"
