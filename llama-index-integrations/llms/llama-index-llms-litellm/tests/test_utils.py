from llama_index.llms.litellm.utils import (
    openai_modelname_to_contextsize,
    to_openai_message_dicts,
    update_tool_calls,
    to_openailike_message_dict,
    from_openai_message_dict,
    from_litellm_message,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
    AudioBlock,
)
from litellm.types.utils import ChatCompletionDeltaToolCall
import json
from unittest.mock import MagicMock
import base64
import io


def test_model_context_size():
    assert openai_modelname_to_contextsize("gpt-4") == 4096
    assert openai_modelname_to_contextsize("gpt-3.5-turbo") == 4096
    assert openai_modelname_to_contextsize("unknown-model") == 2048


def test_message_conversion():
    # Test converting to OpenAI message format
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
        ChatMessage(role=MessageRole.SYSTEM, content="Be helpful"),
    ]

    openai_messages = to_openai_message_dicts(messages)
    assert len(openai_messages) == 3
    assert openai_messages[0]["role"] == "user"
    assert openai_messages[0]["content"] == "Hello"
    assert openai_messages[1]["role"] == "assistant"
    assert openai_messages[2]["role"] == "system"


def test_single_text_block_conversion():
    openai_message = to_openailike_message_dict(
        ChatMessage(role=MessageRole.USER, content=[TextBlock(text="Hello, world!")])
    )

    # Since this is just a text block, it should be simplified to a string
    assert openai_message["role"] == "user"
    assert openai_message["content"] == "Hello, world!"


def test_multiple_text_block_conversion():
    openai_message = to_openailike_message_dict(
        ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text="Hello"), TextBlock(text=", world!")],
        )
    )
    # Content should be concatenated if only text blocks
    assert openai_message["content"] == "Hello, world!"


def test_image_block_url_conversion():
    image_url = "https://example.com/image.jpg"
    openai_message = to_openailike_message_dict(
        ChatMessage(
            role=MessageRole.USER,
            content=[
                ImageBlock(url=image_url),
                TextBlock(text="What's in this image?"),
            ],
        )
    )

    # should have both block types
    assert openai_message["role"] == "user"
    assert isinstance(openai_message["content"], list)
    assert len(openai_message["content"]) == 2

    image_content = openai_message["content"][0]
    assert image_content["type"] == "image_url"
    assert image_content["image_url"]["url"] == image_url
    assert image_content["image_url"]["detail"] == "auto"

    text_content = openai_message["content"][1]
    assert text_content["type"] == "text"
    assert text_content["text"] == "What's in this image?"


def test_image_block_detail_conversion():
    # Test with custom detail level
    image_url = "https://example.com/image.jpg"
    image_block = ImageBlock(url=image_url, detail="high")
    openai_message = to_openailike_message_dict(
        ChatMessage(role=MessageRole.USER, content=[image_block])
    )
    assert openai_message["content"][0]["image_url"]["detail"] == "high"


def test_image_block_binary_data():
    # Mock binary image data
    mock_image_data = b"fake_image_data"

    openai_message = to_openailike_message_dict(
        ChatMessage(
            role=MessageRole.USER,
            content=[ImageBlock(image=mock_image_data, image_mimetype="image/jpeg")],
        )
    )

    # Verify correct format for image block with binary data
    assert openai_message["role"] == "user"
    assert isinstance(openai_message["content"], list)
    assert len(openai_message["content"]) == 1

    image_content = openai_message["content"][0]
    assert image_content["type"] == "image_url"
    assert image_content["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image_content["image_url"]["detail"] == "auto"


def test_image_block_file_data(monkeypatch):
    # Create a mock for resolve_image that returns base64 encoded data
    mock_image_data = b"fake_image_data"

    encoded_data = base64.b64encode(mock_image_data)
    mock_file = MagicMock()
    mock_file.read.return_value = encoded_data

    # Set up the monkeypatch to replace the resolve_image method
    def mock_resolve_image(*args, **kwargs):
        return mock_file

    monkeypatch.setattr(ImageBlock, "resolve_image", mock_resolve_image)

    # Create ImageBlock with file path
    image_block = ImageBlock(
        file_path="/path/to/image.jpg", image_mimetype="image/jpeg"
    )
    message = ChatMessage(role=MessageRole.USER, content=[image_block])

    # Convert to OpenAI format
    openai_message = to_openailike_message_dict(message)

    # Verify correct format for image block with file data
    assert openai_message["role"] == "user"
    assert isinstance(openai_message["content"], list)
    assert len(openai_message["content"]) == 1

    image_content = openai_message["content"][0]
    assert image_content["type"] == "image_url"
    assert "data:image/jpeg;base64," in image_content["image_url"]["url"]
    assert image_content["image_url"]["detail"] == "auto"


def test_audio_block_conversion(monkeypatch):
    """Test converting a ChatMessage with AudioBlock to OpenAI format."""
    # Mock binary audio data
    mock_audio_data = b"fake_audio_data"
    mock_audio_file = io.BytesIO(mock_audio_data)

    # Create a mock for resolve_audio that returns base64 encoded data
    encoded_data = base64.b64encode(mock_audio_data)
    mock_file = MagicMock()
    mock_file.read.return_value = encoded_data

    # Set up the monkeypatch to replace the resolve_audio method
    def mock_resolve_audio(*args, **kwargs):
        return mock_file

    monkeypatch.setattr(AudioBlock, "resolve_audio", mock_resolve_audio)

    # Create AudioBlock
    audio_block = AudioBlock(audio_data=mock_audio_file, format="mp3")
    message = ChatMessage(role=MessageRole.USER, content=[audio_block])

    # Convert to OpenAI format
    openai_message = to_openailike_message_dict(message)

    # Verify correct format for audio block
    assert openai_message["role"] == "user"
    assert isinstance(openai_message["content"], list)
    assert len(openai_message["content"]) == 1

    audio_content = openai_message["content"][0]
    assert audio_content["type"] == "input_audio"
    assert "data" in audio_content["input_audio"]
    assert audio_content["input_audio"]["format"] == "mp3"


def test_mixed_content_conversion():
    """Test converting a ChatMessage with mixed content blocks to OpenAI format."""
    # Create various blocks
    text_block = TextBlock(text="This is an image:")

    # URL image
    image_url = "https://example.com/image.jpg"
    image_block = ImageBlock(url=image_url)

    # Create message with mixed content
    message = ChatMessage(role=MessageRole.USER, content=[text_block, image_block])

    # Convert to OpenAI format
    openai_message = to_openailike_message_dict(message)

    # Verify correct format for mixed content
    assert openai_message["role"] == "user"
    assert isinstance(openai_message["content"], list)
    assert len(openai_message["content"]) == 2

    text_content = openai_message["content"][0]
    assert text_content["type"] == "text"
    assert text_content["text"] == "This is an image:"

    image_content = openai_message["content"][1]
    assert image_content["type"] == "image_url"
    assert image_content["image_url"]["url"] == image_url


def test_from_openai_message_dict():
    """Test converting OpenAI message dict back to ChatMessage."""
    # Simple text message
    openai_message = {"role": "user", "content": "Hello, world!"}
    chat_message = from_openai_message_dict(openai_message)
    assert chat_message.role == MessageRole.USER
    assert chat_message.content == "Hello, world!"

    # Message with tool_calls
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
        }
    ]
    openai_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    chat_message = from_openai_message_dict(openai_message)
    assert chat_message.role == MessageRole.ASSISTANT
    assert chat_message.content is None
    assert "tool_calls" in chat_message.additional_kwargs
    assert chat_message.additional_kwargs["tool_calls"] == tool_calls


def test_from_litellm_message():
    """Test converting litellm.utils.Message to ChatMessage."""
    # Simple text message
    litellm_message = {"role": "user", "content": "Hello, world!"}
    chat_message = from_litellm_message(litellm_message)
    assert chat_message.role == "user"
    assert chat_message.content == "Hello, world!"

    # Message with tool_calls
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
        }
    ]
    litellm_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    chat_message = from_litellm_message(litellm_message)
    assert chat_message.role == "assistant"
    assert chat_message.content is None
    assert "tool_calls" in chat_message.additional_kwargs
    assert chat_message.additional_kwargs["tool_calls"] == tool_calls


def test_update_tool_calls_empty_list():
    """Test updating an empty tool_calls list with a new delta."""
    tool_calls = []
    delta = ChatCompletionDeltaToolCall(
        index=0,
        id="call_123",
        type="function",
        function={"name": "test_function", "arguments": ""},
    )

    # Update and verify a new tool call is added
    result = update_tool_calls(tool_calls, [delta])
    assert len(result) == 1
    assert result[0]["id"] == "call_123"
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "test_function"


def test_update_tool_calls_partial_arguments():
    """Test partial update to tool call arguments."""
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "index": 0,
            "function": {"name": "test_function", "arguments": "{"},
        }
    ]

    delta = ChatCompletionDeltaToolCall(index=0, function={"arguments": '"key":'})

    result = update_tool_calls(tool_calls, [delta])
    assert result[0]["function"]["arguments"] == '{"key":'


def test_update_tool_calls_multiple_parallel():
    """Test handling multiple parallel tool calls."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "index": 0,
            "function": {"name": "func1", "arguments": "{"},
        },
        {
            "id": "call_2",
            "type": "function",
            "index": 1,
            "function": {"name": "func2", "arguments": "{"},
        },
    ]

    deltas = [
        ChatCompletionDeltaToolCall(index=0, function={"arguments": '"a":1}'}),
        ChatCompletionDeltaToolCall(index=1, function={"arguments": '"b":2}'}),
    ]

    result = update_tool_calls(tool_calls, deltas)
    assert len(result) == 2
    assert result[0]["function"]["arguments"] == '{"a":1}'
    assert result[1]["function"]["arguments"] == '{"b":2}'


def test_update_tool_calls_id_only():
    """Test delta with no function data, just ID."""
    tool_calls = []
    delta = ChatCompletionDeltaToolCall(
        index=0, id="call_empty", function={"arguments": ""}
    )

    result = update_tool_calls(tool_calls, [delta])
    assert len(result) == 1
    assert result[0]["id"] == "call_empty"
    assert "function" in result[0]


def test_update_tool_calls_incremental_building():
    """Test incremental building of function name and arguments."""
    tool_calls = [
        {
            "id": "call_inc",
            "type": "function",
            "index": 0,
            "function": {"name": "get_", "arguments": ""},
        }
    ]

    # First update: add to function name
    delta1 = ChatCompletionDeltaToolCall(
        index=0, id="call_inc", function={"name": "weather", "arguments": ""}
    )
    result = update_tool_calls(tool_calls, [delta1])
    assert result[0]["function"]["name"] == "get_weather"

    # Second update: start adding arguments
    delta2 = ChatCompletionDeltaToolCall(
        index=0, id="call_inc", function={"arguments": '{"loc', "name": None}
    )
    result = update_tool_calls(result, [delta2])

    # Third update: continue building arguments
    delta3 = ChatCompletionDeltaToolCall(
        index=0, id="call_inc", function={"arguments": 'ation":"', "name": None}
    )
    result = update_tool_calls(result, [delta3])

    # Final update: complete arguments
    delta4 = ChatCompletionDeltaToolCall(
        index=0, id="call_inc", function={"arguments": 'NYC"}', "name": None}
    )
    result = update_tool_calls(result, [delta4])

    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["arguments"] == '{"location":"NYC"}'

    # Verify valid JSON object
    args = json.loads(result[0]["function"]["arguments"])
    assert args["location"] == "NYC"
