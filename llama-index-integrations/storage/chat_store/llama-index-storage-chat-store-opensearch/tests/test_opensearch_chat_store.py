"""Tests for OpensearchChatStore."""

from typing import Any, Dict
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.storage.chat_store.opensearch.base import (
    OpensearchChatStore,
    _message_to_str,
    _str_to_message,
)


# ---- serialisation helpers ----


def test_message_roundtrip() -> None:
    """ChatMessage survives a serialise/deserialise cycle."""
    msg = ChatMessage(role=MessageRole.USER, content="hello")
    assert _str_to_message(_message_to_str(msg)).content == "hello"


# ---- fixtures ----


def _make_hit(session_id: str, idx: int, message: ChatMessage) -> Dict[str, Any]:
    """Build a fake OpenSearch hit dict."""
    return {
        "_id": f"{session_id}_{idx}",
        "_source": {
            "session_id": session_id,
            "index": idx,
            "message": _message_to_str(message),
        },
    }


def _msg(content: str, role: MessageRole = MessageRole.USER) -> ChatMessage:
    return ChatMessage(role=role, content=content)


@pytest.fixture()
def mock_os_client() -> MagicMock:
    client = MagicMock()
    client.indices.exists.return_value = True
    return client


@pytest.fixture()
def mock_async_client() -> AsyncMock:
    client = AsyncMock()
    client.indices.exists.return_value = True
    return client


@pytest.fixture()
def chat_store(
    mock_os_client: MagicMock, mock_async_client: AsyncMock
) -> OpensearchChatStore:
    with (
        patch(
            "llama_index.storage.chat_store.opensearch.base.OpenSearch",
            return_value=mock_os_client,
        ),
        patch(
            "llama_index.storage.chat_store.opensearch.base.AsyncOpenSearch",
            return_value=mock_async_client,
        ),
    ):
        return OpensearchChatStore(
            opensearch_url="https://localhost:9200",
            index="test_chat",
        )


# ---- class_name ----


def test_class_name() -> None:
    assert OpensearchChatStore.class_name() == "OpensearchChatStore"


# ---- set_messages / get_messages ----


def test_set_messages(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    messages = [_msg("hi"), _msg("bye")]
    chat_store.set_messages("s1", messages)

    # Should delete existing and index new docs
    mock_os_client.delete_by_query.assert_called_once()
    assert mock_os_client.index.call_count == 2


def test_get_messages(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    m1, m2 = _msg("first"), _msg("second")
    mock_os_client.search.return_value = {
        "hits": {"hits": [_make_hit("s1", 0, m1), _make_hit("s1", 1, m2)]}
    }

    result = chat_store.get_messages("s1")
    assert len(result) == 2
    assert result[0].content == "first"
    assert result[1].content == "second"


# ---- add_message ----


def test_add_message_append(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    """add_message without idx appends at the end."""
    # _get_next_index does a search for highest index
    mock_os_client.search.return_value = {
        "hits": {"hits": [_make_hit("s1", 2, _msg("existing"))]}
    }

    chat_store.add_message("s1", _msg("new"))
    # Should index at position 3
    call_kwargs = mock_os_client.index.call_args
    assert call_kwargs.kwargs["body"]["index"] == 3


def test_add_message_insert(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    """add_message with idx shifts existing messages."""
    existing = _make_hit("s1", 1, _msg("old"))
    mock_os_client.search.return_value = {"hits": {"hits": [existing]}}

    chat_store.add_message("s1", _msg("inserted"), idx=1)

    # Should update the existing doc's index to 2
    mock_os_client.update.assert_called_once()
    update_body = mock_os_client.update.call_args.kwargs["body"]
    assert update_body == {"doc": {"index": 2}}

    # Should index the new message at position 1
    index_body = mock_os_client.index.call_args.kwargs["body"]
    assert index_body["index"] == 1


# ---- delete_messages ----


def test_delete_messages(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    m1 = _msg("to-delete")
    mock_os_client.search.return_value = {"hits": {"hits": [_make_hit("s1", 0, m1)]}}

    result = chat_store.delete_messages("s1")
    assert result is not None
    assert len(result) == 1
    assert result[0].content == "to-delete"
    mock_os_client.delete_by_query.assert_called()


def test_delete_messages_empty(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    result = chat_store.delete_messages("empty")
    assert result is None


# ---- delete_message ----


def test_delete_message(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    m = _msg("target")
    # First search: find the message to delete
    # Second search (reindex): return remaining messages
    mock_os_client.search.side_effect = [
        {"hits": {"hits": [_make_hit("s1", 1, m)]}},
        # reindex: session_sorted_query returns remaining
        {"hits": {"hits": [_make_hit("s1", 0, _msg("kept"))]}},
    ]

    result = chat_store.delete_message("s1", 1)
    assert result is not None
    assert result.content == "target"
    mock_os_client.delete.assert_called_once()


def test_delete_message_not_found(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    result = chat_store.delete_message("s1", 99)
    assert result is None


# ---- delete_last_message ----


def test_delete_last_message(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    m = _msg("last")
    mock_os_client.search.return_value = {"hits": {"hits": [_make_hit("s1", 5, m)]}}

    result = chat_store.delete_last_message("s1")
    assert result is not None
    assert result.content == "last"
    mock_os_client.delete.assert_called_once()


def test_delete_last_message_empty(
    chat_store: OpensearchChatStore, mock_os_client: MagicMock
) -> None:
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    result = chat_store.delete_last_message("s1")
    assert result is None


# ---- get_keys ----


def test_get_keys(chat_store: OpensearchChatStore, mock_os_client: MagicMock) -> None:
    mock_os_client.search.return_value = {
        "aggregations": {"unique_sessions": {"buckets": [{"key": "s1"}, {"key": "s2"}]}}
    }

    keys = chat_store.get_keys()
    assert keys == ["s1", "s2"]


# ---- async tests ----


@pytest.mark.asyncio
async def test_aget_messages(
    chat_store: OpensearchChatStore, mock_async_client: AsyncMock
) -> None:
    m1, m2 = _msg("a"), _msg("b")
    mock_async_client.search.return_value = {
        "hits": {"hits": [_make_hit("s1", 0, m1), _make_hit("s1", 1, m2)]}
    }

    result = await chat_store.aget_messages("s1")
    assert len(result) == 2
    assert result[0].content == "a"


@pytest.mark.asyncio
async def test_aset_messages(
    chat_store: OpensearchChatStore, mock_async_client: AsyncMock
) -> None:
    messages = [_msg("x"), _msg("y")]
    await chat_store.aset_messages("s1", messages)

    mock_async_client.delete_by_query.assert_called_once()
    assert mock_async_client.index.call_count == 2


@pytest.mark.asyncio
async def test_adelete_last_message(
    chat_store: OpensearchChatStore, mock_async_client: AsyncMock
) -> None:
    m = _msg("async-last")
    mock_async_client.search.return_value = {"hits": {"hits": [_make_hit("s1", 3, m)]}}

    result = await chat_store.adelete_last_message("s1")
    assert result is not None
    assert result.content == "async-last"
    mock_async_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_aget_keys(
    chat_store: OpensearchChatStore, mock_async_client: AsyncMock
) -> None:
    mock_async_client.search.return_value = {
        "aggregations": {
            "unique_sessions": {"buckets": [{"key": "a"}, {"key": "b"}, {"key": "c"}]}
        }
    }

    keys = await chat_store.aget_keys()
    assert keys == ["a", "b", "c"]
