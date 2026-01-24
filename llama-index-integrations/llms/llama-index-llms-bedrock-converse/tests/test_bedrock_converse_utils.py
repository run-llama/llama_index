from io import BytesIO
from typing import Literal
from unittest.mock import MagicMock, patch

import aioboto3
import pytest
from botocore.config import Config
from llama_index.core.base.llms.types import (
    AudioBlock,
    CacheControl,
    CachePoint,
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.bedrock_converse.utils import (
    __get_img_format_from_image_mimetype,
    _content_block_to_bedrock_format,
    converse_with_retry,
    converse_with_retry_async,
    get_model_name,
    messages_to_converse_messages,
    tools_to_converse_tools,
)

EXP_RESPONSE = "Test"
EXP_STREAM_RESPONSE = ["Test ", "value"]


class MockExceptions:
    class ThrottlingException(Exception):
        pass


class AsyncMockClient:
    def __init__(self) -> None:
        self.exceptions = MockExceptions()

    async def __aenter__(self) -> "AsyncMockClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": EXP_RESPONSE}]}}}

    async def converse_stream(self, *args, **kwargs):
        async def stream_generator():
            for element in EXP_STREAM_RESPONSE:
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": element},
                        "contentBlockIndex": 0,
                    }
                }
            # Add messageStop and metadata events for token usage testing
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {
                "metadata": {
                    "usage": {"inputTokens": 15, "outputTokens": 26, "totalTokens": 41},
                    "metrics": {"latencyMs": 886},
                }
            }

        return {"stream": stream_generator()}


class MockAsyncSession:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def client(self, *args, **kwargs):
        return AsyncMockClient()


@pytest.fixture()
def mock_aioboto3_session(monkeypatch):
    monkeypatch.setattr("aioboto3.Session", MockAsyncSession)


def test_get_model_name_translates_us():
    assert (
        get_model_name("us.meta.llama3-2-3b-instruct-v1:0")
        == "meta.llama3-2-3b-instruct-v1:0"
    )


def test_get_model_name_translates_global():
    assert (
        get_model_name("global.anthropic.claude-sonnet-4-5-20250929-v1:0")
        == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    )


def test_get_model_name_translates_jp():
    assert (
        get_model_name("jp.anthropic.claude-sonnet-4-5-20250929-v1:0")
        == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    )


def test_get_model_name_does_nottranslate_cn():
    assert (
        get_model_name("cn.meta.llama3-2-3b-instruct-v1:0")
        == "cn.meta.llama3-2-3b-instruct-v1:0"
    )


def test_get_model_name_does_nottranslate_unsupported():
    assert get_model_name("cohere.command-r-plus-v1:0") == "cohere.command-r-plus-v1:0"


def test_get_model_name_throws_inference_profile_exception():
    with pytest.raises(ValueError):
        assert get_model_name("us.cohere.command-r-plus-v1:0")


def test_get_img_format_jpeg():
    assert __get_img_format_from_image_mimetype("image/jpeg") == "jpeg"


def test_get_img_format_png():
    assert __get_img_format_from_image_mimetype("image/png") == "png"


def test_get_img_format_gif():
    assert __get_img_format_from_image_mimetype("image/gif") == "gif"


def test_get_img_format_webp():
    assert __get_img_format_from_image_mimetype("image/webp") == "webp"


def test_get_img_format_unsupported(caplog):
    result = __get_img_format_from_image_mimetype("image/unsupported")
    assert result == "png"
    assert "Unsupported image mimetype" in caplog.text


def test_content_block_to_bedrock_format_text():
    text_block = TextBlock(text="Hello, world!")
    result = _content_block_to_bedrock_format(text_block, MessageRole.USER)
    assert result == {"text": "Hello, world!"}


def test_content_block_to_bedrock_format_thinking():
    think_block = ThinkingBlock(content="Hello, world!")
    result = _content_block_to_bedrock_format(think_block, MessageRole.USER)
    assert result == {"reasoningContent": {"reasoningText": {"text": "Hello, world!"}}}


def test_cache_point_block():
    cache_point = CachePoint(cache_control=CacheControl(type="default"))
    result = _content_block_to_bedrock_format(cache_point, MessageRole.USER)
    assert result == {"cachePoint": {"type": "default"}}
    cache_point1 = CachePoint(cache_control=CacheControl(type="persistent"))
    result1 = _content_block_to_bedrock_format(cache_point1, MessageRole.USER)
    assert result1 == {"cachePoint": {"type": "default"}}


@patch("llama_index.core.base.llms.types.ImageBlock.resolve_image")
def test_content_block_to_bedrock_format_image_user(mock_resolve):
    mock_bytes = BytesIO(b"fake_image_data")
    mock_bytes.read = MagicMock(return_value=b"fake_image_data")
    mock_resolve.return_value = mock_bytes

    image_block = ImageBlock(image=b"", image_mimetype="image/png")

    result = _content_block_to_bedrock_format(image_block, MessageRole.USER)

    assert "image" in result
    assert result["image"]["format"] == "png"
    assert "bytes" in result["image"]["source"]
    mock_resolve.assert_called_once()


@patch("llama_index.core.base.llms.types.ImageBlock.resolve_image")
def test_content_block_to_bedrock_format_image_assistant(mock_resolve, caplog):
    image_block = ImageBlock(image=b"", image_mimetype="image/png")
    result = _content_block_to_bedrock_format(image_block, MessageRole.ASSISTANT)

    assert result is None
    assert "only supports image blocks for user messages" in caplog.text
    mock_resolve.assert_not_called()


def test_content_block_to_bedrock_format_audio(caplog):
    audio_block = AudioBlock(audio=b"test_audio")
    result = _content_block_to_bedrock_format(audio_block, MessageRole.USER)

    assert result is None
    assert "Audio blocks are not supported" in caplog.text


def test_content_block_to_bedrock_format_unsupported(caplog):
    unsupported_block = object()
    result = _content_block_to_bedrock_format(unsupported_block, MessageRole.USER)

    assert result is None
    assert "Unsupported block type" in caplog.text
    assert str(type(unsupported_block)) in caplog.text


def test_tools_to_converse_tools_with_tool_required():
    """Test that tool_required=True sets toolChoice to {"any": {}}."""

    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    result = tools_to_converse_tools([tool], tool_required=True)

    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["toolSpec"]["name"] == "search_tool"
    assert result["toolChoice"] == {"any": {}}


def test_tools_to_converse_tools_without_tool_required():
    """Test that tool_required=False sets toolChoice to {"auto": {}}."""

    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    result = tools_to_converse_tools([tool], tool_required=False)

    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["toolSpec"]["name"] == "search_tool"
    assert result["toolChoice"] == {"auto": {}}


def test_tools_to_converse_tools_with_custom_tool_choice():
    """Test that a custom tool_choice overrides tool_required."""

    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    custom_tool_choice = {"specific": {"name": "search_tool"}}
    result = tools_to_converse_tools(
        [tool], tool_choice=custom_tool_choice, tool_required=True
    )

    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["toolSpec"]["name"] == "search_tool"
    assert result["toolChoice"] == custom_tool_choice


def test_tools_to_converse_tools_with_cache_enabled():
    """Test that cachePoint is configured when setting tool_caching=True"""

    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    result = tools_to_converse_tools([tool], tool_caching=True)

    assert "tools" in result
    assert len(result["tools"]) == 2
    assert result["tools"][0]["toolSpec"]["name"] == "search_tool"
    assert result["tools"][1]["cachePoint"]["type"] == "default"


# Tests for messages_to_converse_messages function
def test_messages_to_converse_messages_simple_user_message():
    """Test converting a simple user message."""
    messages = [ChatMessage(role=MessageRole.USER, content="Hello, world!")]

    converse_messages, system_prompt = messages_to_converse_messages(messages)

    assert len(converse_messages) == 1
    assert converse_messages[0]["role"] == "user"
    assert converse_messages[0]["content"] == [{"text": "Hello, world!"}]
    assert system_prompt == []


def test_messages_to_converse_messages_with_system_prompt():
    """Test converting messages with a system prompt."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]

    converse_messages, system_prompt = messages_to_converse_messages(messages)

    assert len(converse_messages) == 1
    assert converse_messages[0]["role"] == "user"
    assert converse_messages[0]["content"] == [{"text": "Hello!"}]
    assert len(system_prompt) == 1
    # System prompt should contain exactly the content we provided, no duplication
    assert system_prompt[0]["text"] == "You are a helpful assistant."
    # Ensure no duplication - content should not appear twice
    system_text = system_prompt[0]["text"]
    assert system_text.count("You are a helpful assistant.") == 1


def test_messages_to_converse_messages_with_cache_point_supported_model():
    """Test cache point handling with a model that supports caching."""
    cache_control = CacheControl(type="default")
    cache_point = CachePoint(cache_control=cache_control)

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text="System context part 1"),
                cache_point,
                TextBlock(text="System context part 2"),
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]

    # Use a model that supports caching
    converse_messages, system_prompt = messages_to_converse_messages(
        messages, model="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    assert len(converse_messages) == 1
    assert converse_messages[0]["role"] == "user"
    # Should produce 3 parts: text + cache_point + text
    assert len(system_prompt) == 3
    assert "System context part 1" in system_prompt[0]["text"]
    assert system_prompt[1]["cachePoint"]["type"] == "default"
    assert "System context part 2" in system_prompt[2]["text"]

    # Verify no duplication of content
    assert system_prompt[0]["text"].count("System context part 1") == 1
    assert system_prompt[2]["text"].count("System context part 2") == 1

    # Verify total input vs output consistency
    input_system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
    assert len(input_system_messages) == 1  # We provided 1 system message


def test_messages_to_converse_messages_with_cache_point_unsupported_model(caplog):
    """Test cache point handling with a model that doesn't support caching."""
    cache_control = CacheControl(type="default")
    cache_point = CachePoint(cache_control=cache_control)

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text="System context part 1"),
                cache_point,
                TextBlock(text="System context part 2"),
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]

    # Use a model that doesn't support caching
    converse_messages, system_prompt = messages_to_converse_messages(
        messages, model="meta.llama3-1-70b-instruct-v1:0"
    )

    assert len(converse_messages) == 1
    assert len(system_prompt) == 2  # Cache point should be omitted
    assert "System context part 1" in system_prompt[0]["text"]
    assert "System context part 2" in system_prompt[1]["text"]
    # Check that warning was logged
    assert "does not support prompt caching" in caplog.text

    # Verify no duplication of content
    assert system_prompt[0]["text"].count("System context part 1") == 1
    assert system_prompt[1]["text"].count("System context part 2") == 1

    # Verify total input vs output consistency
    input_system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
    assert len(input_system_messages) == 1  # We provided 1 system message


def test_messages_to_converse_messages_with_cache_point_no_model():
    """Test cache point handling when no model is specified (should include cache point)."""
    cache_control = CacheControl(type="default")
    cache_point = CachePoint(cache_control=cache_control)

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text="System context"),
                cache_point,
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]

    # No model specified - should include cache point
    converse_messages, system_prompt = messages_to_converse_messages(messages)

    assert "System context" in system_prompt[0]["text"]
    assert system_prompt[1]["cachePoint"]["type"] == "default"

    # Verify no duplication of content
    assert system_prompt[0]["text"].count("System context") == 1

    # Verify total input vs output consistency
    input_system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
    assert len(input_system_messages) == 1  # We provided 1 system message

    # Should produce 2 parts: text + cache_point
    assert len(system_prompt) == 2


def test_messages_to_converse_messages_mixed_system_content():
    """Test system messages with both string content and blocks."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="String system prompt"),
        ChatMessage(
            role=MessageRole.SYSTEM, blocks=[TextBlock(text="Block system prompt")]
        ),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]

    converse_messages, system_prompt = messages_to_converse_messages(messages)

    # Both system prompts should be merged into a single message
    system_text = system_prompt[0]["text"]
    assert "String system prompt" in system_text
    assert "Block system prompt" in system_text
    # Verify total number of input vs output messages
    input_system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
    assert len(input_system_messages) == 2  # We provided 2 system messages
    # But they should be combined into 1 system prompt
    assert len(system_prompt) == 1
    # Ensure no duplication - each piece of content appears only once
    assert system_text.count("String system prompt") == 1
    assert system_text.count("Block system prompt") == 1


def test_messages_to_converse_messages_empty_text_blocks():
    """Test handling of empty text blocks."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text=""),  # Empty text block
                TextBlock(text="Hello!"),
            ],
        )
    ]

    converse_messages, system_prompt = messages_to_converse_messages(messages)

    assert len(converse_messages) == 1
    # Only non-empty text block should be included
    assert len(converse_messages[0]["content"]) == 1
    assert converse_messages[0]["content"][0]["text"] == "Hello!"


def test_messages_to_converse_messages_tool_calls():
    """Test handling of tool calls in messages."""
    messages = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I'll search for that information.",
            additional_kwargs={
                "tool_calls": [
                    {
                        "toolUseId": "tool_123",
                        "name": "search",
                        "input": {"query": "test query"},
                    }
                ]
            },
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content="Search results here",
            additional_kwargs={"tool_call_id": "tool_123"},
        ),
    ]

    converse_messages, system_prompt = messages_to_converse_messages(messages)

    # Tool calls are combined with the assistant message content in current implementation
    assert (
        len(converse_messages) == 2
    )  # assistant message (with both text and tool call), tool result

    # Check assistant message (contains both text and tool call)
    assert converse_messages[0]["role"] == "assistant"
    assert len(converse_messages[0]["content"]) == 2  # text + tool call
    assert (
        converse_messages[0]["content"][0]["text"]
        == "I'll search for that information."
    )
    assert "toolUse" in converse_messages[0]["content"][1]
    assert converse_messages[0]["content"][1]["toolUse"]["toolUseId"] == "tool_123"
    assert converse_messages[0]["content"][1]["toolUse"]["name"] == "search"

    # Check tool result
    assert (
        converse_messages[1]["role"] == "user"
    )  # Bedrock requires tool results as user role
    assert "toolResult" in converse_messages[1]["content"][0]
    assert converse_messages[1]["content"][0]["toolResult"]["toolUseId"] == "tool_123"


# Tests for converse_with_retry function
class MockClient:
    def __init__(self):
        self.exceptions = MagicMock()
        self.exceptions.ThrottlingException = Exception

    def converse(self, **kwargs):
        return {"output": {"message": {"content": [{"text": "Test response"}]}}}

    def converse_stream(self, **kwargs):
        def stream_generator():
            yield {
                "contentBlockDelta": {
                    "delta": {"text": "Test "},
                    "contentBlockIndex": 0,
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {"text": "stream"},
                    "contentBlockIndex": 0,
                }
            }

        return {"stream": stream_generator()}


def test_converse_with_retry_string_system_prompt():
    """Test converse_with_retry with string system prompt."""
    client = MockClient()
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Mock the converse method to capture the kwargs
    original_converse = client.converse
    captured_kwargs = {}

    def mock_converse(**kwargs):
        captured_kwargs.update(kwargs)
        return original_converse(**kwargs)

    client.converse = mock_converse

    response = converse_with_retry(
        client=client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        system_prompt="You are a helpful assistant.",
        max_retries=1,
        stream=False,
    )

    assert response is not None
    assert "system" in captured_kwargs
    assert captured_kwargs["system"] == [{"text": "You are a helpful assistant."}]


def test_converse_with_retry_list_system_prompt():
    """Test converse_with_retry with list system prompt."""
    client = MockClient()
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Mock the converse method to capture the kwargs
    original_converse = client.converse
    captured_kwargs = {}

    def mock_converse(**kwargs):
        captured_kwargs.update(kwargs)
        return original_converse(**kwargs)

    client.converse = mock_converse

    system_prompt = [
        {"text": "You are a helpful assistant."},
        {"cachePoint": {"type": "default"}},
        {"text": "Additional context."},
    ]

    response = converse_with_retry(
        client=client,
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        system_prompt=system_prompt,
        max_retries=1,
        stream=False,
    )

    assert response is not None
    assert "system" in captured_kwargs
    assert captured_kwargs["system"] == system_prompt


@pytest.mark.parametrize("stream_processing_mode", ["sync", "async"])
def test_converse_with_retry_guardrail_stream_processing_mode(
    stream_processing_mode: Literal["sync", "async"],
):
    """
    Test use of guardrail_stream_processing_mode in converse_with_retry with streaming.
    """
    client = MockClient()

    with patch.object(client, "converse_stream") as patched_converse_stream:
        converse_with_retry(
            client=client,
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[],
            stream=True,  # with streaming
            guardrail_identifier="IDENT",
            guardrail_version="DRAFT",
            guardrail_stream_processing_mode=stream_processing_mode,
        )
        call_kwargs = patched_converse_stream.call_args.kwargs
        assert "guardrailConfig" in call_kwargs
        assert "streamProcessingMode" in call_kwargs["guardrailConfig"]
        assert (
            call_kwargs["guardrailConfig"]["streamProcessingMode"]
            == stream_processing_mode
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_processing_mode", ["sync", "async"])
async def test_converse_with_retry_async_guardrail_stream_processing_mode(
    stream_processing_mode: Literal["sync", "async"],
    mock_aioboto3_session,
):
    """
    Test use of guardrail_stream_processing_mode in converse_with_retry_async with streaming.
    """
    session = aioboto3.Session()
    client = AsyncMockClient()

    with patch.object(
        AsyncMockClient, "converse_stream", wraps=client.converse_stream
    ) as patched_converse_stream:
        response_gen = await converse_with_retry_async(
            session=session,
            config=Config(),
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[],
            stream=True,  # with streaming
            guardrail_identifier="IDENT",
            guardrail_version="DRAFT",
            guardrail_stream_processing_mode=stream_processing_mode,
        )
        async for _ in response_gen:
            pass
        call_kwargs = patched_converse_stream.call_args.kwargs
        assert "guardrailConfig" in call_kwargs
        assert "streamProcessingMode" in call_kwargs["guardrailConfig"]
        assert (
            call_kwargs["guardrailConfig"]["streamProcessingMode"]
            == stream_processing_mode
        )


def test_converse_with_retry_guardrail_stream_processing_mode_without_stream():
    """
    Test use of guardrail_stream_processing_mode in converse_with_retry WITHOUT streaming.
    """
    client = MockClient()

    with patch.object(client, "converse") as patched_converse:
        converse_with_retry(
            client=client,
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[],
            stream=False,  # without streaming
            guardrail_identifier="IDENT",
            guardrail_version="DRAFT",
            guardrail_stream_processing_mode="async",
        )
        call_kwargs = patched_converse.call_args.kwargs
        assert "guardrailConfig" in call_kwargs
        assert "streamProcessingMode" not in call_kwargs["guardrailConfig"]


@pytest.mark.asyncio
async def test_converse_with_retry_async_guardrail_stream_processing_mode_without_stream(
    mock_aioboto3_session,
):
    """
    Test use of guardrail_stream_processing_mode in converse_with_retry_async WITHOUT streaming.
    """
    session = aioboto3.Session()
    client = AsyncMockClient()

    with patch.object(
        AsyncMockClient, "converse", wraps=client.converse
    ) as patched_converse:
        await converse_with_retry_async(
            session=session,
            config=Config(),
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[],
            stream=False,  # without streaming
            guardrail_identifier="IDENT",
            guardrail_version="DRAFT",
            guardrail_stream_processing_mode="async",
        )
        call_kwargs = patched_converse.call_args.kwargs
        assert "guardrailConfig" in call_kwargs
        assert "streamProcessingMode" not in call_kwargs["guardrailConfig"]
