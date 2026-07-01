"""Tests for DakeraMemory integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.memory.dakera import DakeraMemory


@pytest.fixture
def memory() -> DakeraMemory:
    return DakeraMemory(
        base_url="http://localhost:8000",
        api_key="test-key",
        session_id="test-session",
        top_k=5,
    )


@pytest.mark.asyncio
async def test_get_returns_messages_on_success(memory: DakeraMemory) -> None:
    """get() returns ChatMessage list when the API returns results."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"content": "User prefers concise answers."},
            {"content": "User is based in France."},
        ]
    }

    with patch.object(memory._client, "post", new=AsyncMock(return_value=mock_response)):
        result = await memory.get(input="What does the user prefer?")

    assert len(result) == 2
    assert all(isinstance(m, ChatMessage) for m in result)
    assert result[0].content == "User prefers concise answers."
    assert result[0].role == MessageRole.SYSTEM


@pytest.mark.asyncio
async def test_get_returns_empty_on_none_input(memory: DakeraMemory) -> None:
    """get() returns empty list when input is None."""
    result = await memory.get(input=None)
    assert result == []


@pytest.mark.asyncio
async def test_get_returns_empty_on_api_error(memory: DakeraMemory) -> None:
    """get() returns empty list gracefully on API failure."""
    with patch.object(
        memory._client, "post", new=AsyncMock(side_effect=httpx.ConnectError("refused"))
    ):
        result = await memory.get(input="something")
    assert result == []


@pytest.mark.asyncio
async def test_put_stores_message(memory: DakeraMemory) -> None:
    """put() calls POST /v1/memories with correct payload."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(memory._client, "post", new=AsyncMock(return_value=mock_response)) as mock_post:
        msg = ChatMessage(role=MessageRole.USER, content="Hello, remember this.")
        await memory.put(msg)

    mock_post.assert_called_once_with(
        "/v1/memories",
        json={"content": "Hello, remember this.", "session_id": "test-session"},
    )


@pytest.mark.asyncio
async def test_put_skips_empty_content(memory: DakeraMemory) -> None:
    """put() does nothing when message content is empty."""
    with patch.object(memory._client, "post", new=AsyncMock()) as mock_post:
        msg = ChatMessage(role=MessageRole.USER, content="")
        await memory.put(msg)

    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_reset_calls_delete(memory: DakeraMemory) -> None:
    """reset() calls DELETE /v1/memories with session_id."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(
        memory._client, "delete", new=AsyncMock(return_value=mock_response)
    ) as mock_delete:
        await memory.reset()

    mock_delete.assert_called_once_with(
        "/v1/memories",
        json={"session_id": "test-session"},
    )


@pytest.mark.asyncio
async def test_set_resets_then_puts_all(memory: DakeraMemory) -> None:
    """set() clears memory then stores each message."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="First message"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Second message"),
    ]

    reset_called = []
    put_calls = []

    async def fake_reset() -> None:
        reset_called.append(True)

    async def fake_put(msg: ChatMessage) -> None:
        put_calls.append(msg.content)

    memory.reset = fake_reset  # type: ignore[method-assign]
    memory.put = fake_put  # type: ignore[method-assign]

    await memory.set(messages)

    assert len(reset_called) == 1
    assert put_calls == ["First message", "Second message"]


@pytest.mark.asyncio
async def test_get_all_broad_search(memory: DakeraMemory) -> None:
    """get_all() sends an empty query to fetch all session memories."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [{"content": "Memory A"}, {"content": "Memory B"}]

    with patch.object(
        memory._client, "post", new=AsyncMock(return_value=mock_response)
    ) as mock_post:
        result = await memory.get_all()

    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["query"] == ""
    assert call_kwargs[1]["json"]["session_id"] == "test-session"
    assert len(result) == 2


def test_class_name() -> None:
    assert DakeraMemory.class_name() == "DakeraMemory"


def test_from_defaults() -> None:
    mem = DakeraMemory.from_defaults(
        base_url="http://localhost:8000",
        api_key="key",
        session_id="s1",
    )
    assert mem.session_id == "s1"
    assert mem.top_k == 10
