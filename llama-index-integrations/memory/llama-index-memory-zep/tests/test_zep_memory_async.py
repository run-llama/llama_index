import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from llama_index.memory.zep import ZepMemory
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# ---------------------------
# Fixtures
# ---------------------------


@pytest.fixture()
def session_and_user_ids():
    return str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture()
def mock_async_zep_client():
    mock_memory = MagicMock()
    mock_memory.get = AsyncMock(return_value=MagicMock(messages=[]))
    mock_memory.add = AsyncMock()
    mock_memory.delete = AsyncMock()
    mock_memory.search_sessions = AsyncMock()

    client = MagicMock()
    client.memory = mock_memory
    return client


@pytest.fixture(autouse=True)
def disable_sync_from_zep(monkeypatch):
    monkeypatch.setattr(ZepMemory, "_sync_from_zep", lambda self: None)


# ---------------------------
# Async Tests
# ---------------------------


@pytest.mark.asyncio()
async def test_zep_memory_from_defaults(mock_async_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids

    memory = ZepMemory.from_defaults(
        zep_client=mock_async_zep_client, session_id=session_id, user_id=user_id
    )

    assert memory.session_id == session_id
    assert memory.user_id == user_id
    assert memory._client == mock_async_zep_client
    assert memory.memory_key == "chat_history"


@pytest.mark.asyncio()
async def test_zep_memory_aget_returns_messages_with_context(
    mock_async_zep_client, session_and_user_ids
):
    session_id, user_id = session_and_user_ids

    mock_async_zep_client.memory.get.return_value = MagicMock(
        messages=[],
        facts=["The user likes cats."],
        summary=MagicMock(content="User enjoys cat memes."),
        context="Recently discussed memes.",
    )

    memory = ZepMemory.from_defaults(mock_async_zep_client, session_id, user_id)

    await memory.aset(
        [
            ChatMessage(role=MessageRole.USER, content="Tell me a meme"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Here's a funny one..."),
        ]
    )

    result = await memory.aget()

    assert result[0].role == MessageRole.SYSTEM
    assert "User enjoys cat memes" in result[0].content
    assert "Recently discussed memes" in result[0].content
    assert result[1].role == MessageRole.USER
    assert result[1].content == "Tell me a meme"


@pytest.mark.asyncio()
async def test_zep_memory_aset_stores_messages(
    mock_async_zep_client, session_and_user_ids
):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_async_zep_client, session_id, user_id)

    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
    ]

    await memory.aset(messages)

    assert memory._primary_memory.get_all() == messages
    mock_async_zep_client.memory.add.assert_awaited_once()


@pytest.mark.asyncio()
async def test_zep_memory_aput(mock_async_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids

    memory = ZepMemory.from_defaults(mock_async_zep_client, session_id, user_id)

    msg = ChatMessage(role=MessageRole.USER, content="What's the weather?")
    await memory.aput(msg)

    all_msgs = await memory.aget_all()
    assert all_msgs[-1].content == "What's the weather?"
    assert all_msgs[-1].role == MessageRole.USER
    mock_async_zep_client.memory.add.assert_called_once()


@pytest.mark.asyncio()
async def test_zep_memory_areset(mock_async_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_async_zep_client, session_id, user_id)

    await memory.areset()
    mock_async_zep_client.memory.delete.assert_awaited_once_with(session_id=session_id)


@pytest.mark.asyncio()
async def test_zep_memory_asearch(mock_async_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_async_zep_client, session_id, user_id)

    mock_async_zep_client.memory.search_sessions.return_value = {
        "results": [{"session_id": "session1"}, {"session_id": "session2"}]
    }

    result = await memory.asearch("search query")

    assert "results" in result
    assert result["results"][0]["session_id"] == "session1"
    assert result["results"][1]["session_id"] == "session2"
