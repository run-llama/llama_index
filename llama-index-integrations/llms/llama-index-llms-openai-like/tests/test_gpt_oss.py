import pytest
from unittest.mock import MagicMock, AsyncMock

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.llms.openai_like.gpt_oss import GptOss

# Define the path to the parent class method we want to mock.
SUPER_ASTREAM_CHAT_PATH = "llama_index.llms.openai_like.base.OpenAILike.astream_chat"


# Helper function to create an async generator from a list
async def list_to_async_generator(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_extracts_thinking_delta(mocker):
    """Test extraction of thinking_delta from the reasoning field."""
    # Arrange
    llm = GptOss(model="test", api_key="fake")
    mock_choice = MagicMock()
    mock_choice.delta.reasoning = "test reasoning"
    response_chunk = ChatResponse(
        message=ChatMessage(role="assistant", content="response"),
        delta="delta",
        raw=MagicMock(choices=[mock_choice]),
    )

    # This mock setup is CORRECT. It simulates a method that must be awaited.
    mock_parent_stream = AsyncMock(
        return_value=list_to_async_generator([response_chunk])
    )
    mocker.patch(SUPER_ASTREAM_CHAT_PATH, mock_parent_stream)

    # Act
    # The fix is here: llm.astream_chat() returns an async generator directly.
    # We iterate over it, we don't await it.
    results = [
        chunk
        async for chunk in llm.astream_chat([ChatMessage(role="user", content="test")])
    ]

    # Assert
    assert len(results) == 1
    assert results[0].additional_kwargs["thinking_delta"] == "test reasoning"
    mock_parent_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_reasoning_field(mocker):
    """Test when the reasoning field doesn't exist."""
    # Arrange
    llm = GptOss(model="test", api_key="fake")
    mock_choice = MagicMock()
    mock_choice.delta = MagicMock(spec=["content"])
    mock_choice.delta.content = "some content"
    response_chunk = ChatResponse(
        message=ChatMessage(role="assistant", content="response"),
        delta="delta",
        raw=MagicMock(choices=[mock_choice]),
        additional_kwargs={"existing": "data"},
    )

    mock_parent_stream = AsyncMock(
        return_value=list_to_async_generator([response_chunk])
    )
    mocker.patch(SUPER_ASTREAM_CHAT_PATH, mock_parent_stream)

    # Act
    results = [
        chunk
        async for chunk in llm.astream_chat([ChatMessage(role="user", content="test")])
    ]

    # Assert
    assert len(results) == 1
    assert "thinking_delta" not in results[0].additional_kwargs
    assert results[0].additional_kwargs == {"existing": "data"}
    mock_parent_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_choices(mocker):
    """Test when the 'choices' array in the raw response is empty."""
    # Arrange
    llm = GptOss(model="test", api_key="fake")
    response_chunk = ChatResponse(
        message=ChatMessage(role="assistant", content="response"),
        delta="delta",
        raw=MagicMock(choices=[]),
        additional_kwargs={},
    )

    mock_parent_stream = AsyncMock(
        return_value=list_to_async_generator([response_chunk])
    )
    mocker.patch(SUPER_ASTREAM_CHAT_PATH, mock_parent_stream)

    # Act
    results = [
        chunk
        async for chunk in llm.astream_chat([ChatMessage(role="user", content="test")])
    ]

    # Assert
    assert len(results) == 1
    assert "thinking_delta" not in results[0].additional_kwargs
    mock_parent_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_choices_attribute(mocker):
    """Test when the raw response object has no 'choices' attribute."""
    # Arrange
    llm = GptOss(model="test", api_key="fake")
    raw_mock = MagicMock(spec=[])
    response_chunk = ChatResponse(
        message=ChatMessage(role="assistant", content="response"),
        delta="delta",
        raw=raw_mock,
        additional_kwargs={},
    )

    mock_parent_stream = AsyncMock(
        return_value=list_to_async_generator([response_chunk])
    )
    mocker.patch(SUPER_ASTREAM_CHAT_PATH, mock_parent_stream)

    # Act
    results = [
        chunk
        async for chunk in llm.astream_chat([ChatMessage(role="user", content="test")])
    ]

    # Assert
    assert len(results) == 1
    assert "thinking_delta" not in results[0].additional_kwargs
    mock_parent_stream.assert_awaited_once()
