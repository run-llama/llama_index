import uuid
from unittest.mock import MagicMock
import pytest
from llama_index.memory.zep import ZepMemory
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# ---------------------------
# Fixtures
# ---------------------------


@pytest.fixture()
def session_and_user_ids():
    return str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture()
def mock_zep_client():
    client = MagicMock()
    client.memory.get.return_value = MagicMock(messages=[])
    return client


# ---------------------------
# Sync Tests
# ---------------------------


def test_zep_memory_from_defaults(mock_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids

    memory = ZepMemory.from_defaults(
        zep_client=mock_zep_client, session_id=session_id, user_id=user_id
    )

    assert memory.session_id == session_id
    assert memory.user_id == user_id
    assert memory._client == mock_zep_client
    assert memory.memory_key == "chat_history"


def test_zep_memory_set_stores_messages(mock_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_zep_client, session_id, user_id)

    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
    ]

    memory.set(messages)

    # Confirm messages are stored in the local buffer
    assert memory._primary_memory.get_all() == messages

    mock_zep_client.memory.add.assert_called_once()


def test_zep_memory_get_returns_messages_with_context(
    mock_zep_client, session_and_user_ids
):
    session_id, user_id = session_and_user_ids

    mock_zep_client.memory.get.return_value = MagicMock(
        messages=[],
        facts=["The user likes cats."],
        summary=MagicMock(content="User enjoys cat memes."),
        context="Recently discussed memes.",
    )

    memory = ZepMemory.from_defaults(mock_zep_client, session_id, user_id)

    memory._primary_memory.set(
        [
            ChatMessage(role=MessageRole.USER, content="Tell me a meme"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Here's a funny one..."),
        ]
    )

    result = memory.get()

    assert result[0].role == MessageRole.SYSTEM
    assert "User enjoys cat memes" in result[0].content
    assert "Recently discussed memes" in result[0].content
    assert result[1].role == MessageRole.USER
    assert result[1].content == "Tell me a meme"


def test_zep_memory_put(mock_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_zep_client, session_id, user_id)

    msg = ChatMessage(role=MessageRole.USER, content="What's the weather?")
    memory.put(msg)

    assert memory._primary_memory.get_all()[-1] == msg
    mock_zep_client.memory.add.assert_called_once()


def test_zep_memory_reset(mock_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_zep_client, session_id, user_id)

    memory.reset()

    memory._primary_memory.reset()
    mock_zep_client.memory.delete.assert_called_once_with(session_id=session_id)


def test_zep_memory_search(mock_zep_client, session_and_user_ids):
    session_id, user_id = session_and_user_ids
    memory = ZepMemory.from_defaults(mock_zep_client, session_id, user_id)

    query = "Tell me a joke"
    memory.search(query)

    mock_zep_client.memory.search_sessions.assert_called_once_with(
        session_ids=[session_id], user_id=user_id, text=query
    )
