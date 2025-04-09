from decimal import Decimal
import httpx
from llama_index.core.base.llms.base import BaseLLM
import pytest
import respx
from unittest.mock import patch
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import (
    MessageRole,
    CompletionResponse,
    ChatResponse,
    TextBlock,
    ImageBlock,
)
import json


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in LiteLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_chat_response(respx_mock)
    message = ChatMessage(role="user", content="Hey! how's it going?")
    chat_response = llm.chat([message])
    assert chat_response.message.blocks[0].text == "Hello, world!"


@pytest.mark.asyncio()
async def test_achat(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_chat_response(respx_mock)
    message = ChatMessage(role="user", content="Hey! how's it going async?")
    chat_response = await llm.achat([message])
    assert chat_response.message.blocks[0].text == "Hello, world!"


def test_completion(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_completion_response(respx_mock)
    response = llm.complete("What is the capital of France?")
    assert isinstance(response, CompletionResponse)
    assert response.text == "Paris is the capital of France."


@pytest.mark.asyncio()
async def test_acompletion(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_completion_response(respx_mock)
    response = await llm.acomplete("What is the capital of France?")
    assert isinstance(response, CompletionResponse)
    assert response.text == "Paris is the capital of France."


def test_stream_chat():
    # Create a mock response that simulates streaming chunks
    def mock_stream():
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="1"), delta="1"
        )
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="12"), delta="2"
        )
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="123"), delta="3"
        )

    # Create the LLM instance
    llm = LiteLLM(model="openai/gpt-fake-model")

    # Mock the stream_chat method
    with patch.object(llm, "_stream_chat", return_value=mock_stream()):
        message = ChatMessage(role="user", content="Count to 3")
        responses = list(llm.stream_chat([message]))

        assert len(responses) == 3
        assert [r.delta for r in responses] == ["1", "2", "3"]
        assert [r.message.content for r in responses] == ["1", "12", "123"]


@pytest.mark.asyncio()
async def test_astream_chat():
    # Create a mock response that simulates streaming chunks
    async def mock_stream():
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="1"), delta="1"
        )
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="12"), delta="2"
        )
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="123"), delta="3"
        )

    # Create the LLM instance
    llm = LiteLLM(model="openai/gpt-fake-model")

    # Mock the astream_chat method
    with patch.object(llm, "_astream_chat", return_value=mock_stream()):
        message = ChatMessage(role="user", content="Count to 3")
        responses = []

        # Get the stream and collect responses
        stream = await llm.astream_chat([message])
        async for response in stream:
            responses.append(response)

        # Verify the responses
        assert len(responses) == 3
        assert [r.delta for r in responses] == ["1", "2", "3"]
        assert [r.message.content for r in responses] == ["1", "12", "123"]


def test_chat_with_system_message(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_chat_response(respx_mock)
    system_message = ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    )
    user_message = ChatMessage(role=MessageRole.USER, content="Hello!")
    chat_response = llm.chat([system_message, user_message])
    assert chat_response.message.blocks[0].text == "Hello, world!"


def add(x: Decimal, y: Decimal) -> Decimal:
    return x + y


def test_tool_calling(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_tool_response(respx_mock)
    message = "what's 1+1?"
    chat_response = llm.chat_with_tools(tools=[add_tool], user_msg=message)
    tool_calls = llm.get_tool_calls_from_response(
        chat_response, error_on_no_tool_call=True
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"x": 1, "y": 1}


@pytest.mark.asyncio()
async def test_achat_tool_calling(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_tool_response(respx_mock)
    message = "what's 1+1?"
    chat_response = await llm.achat_with_tools(tools=[add_tool], user_msg=message)
    tool_calls = llm.get_tool_calls_from_response(
        chat_response, error_on_no_tool_call=True
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"x": 1, "y": 1}


def test_token_calculation_errors():
    # Test case 1: Missing tiktoken
    with patch.dict("sys.modules", {"tiktoken": None}):
        llm = LiteLLM(model="gpt-3.5-turbo")
        with pytest.raises(ImportError) as exc_info:
            llm._get_max_token_for_prompt("test prompt")
        assert "Please install tiktoken" in str(exc_info.value)

    # Test case 2: Prompt too long
    with patch("tiktoken.encoding_for_model") as mock_encoding:
        # Mock the encoding to return a token count that exceeds context window
        mock_encoding.return_value.encode.return_value = [1] * 5000  # 5000 tokens
        llm = LiteLLM(model="gpt-3.5-turbo")
        with pytest.raises(ValueError) as exc_info:
            llm._get_max_token_for_prompt("test prompt")
        assert "The prompt is too long for the model" in str(exc_info.value)

    # Test case 3: Unknown model encoding fallback
    with patch("tiktoken.encoding_for_model") as mock_encoding, patch(
        "tiktoken.get_encoding"
    ) as mock_get_encoding:
        # Mock encoding_for_model to raise KeyError
        mock_encoding.side_effect = KeyError("Unknown model")
        # Mock get_encoding to return a working encoding
        mock_get_encoding.return_value.encode.return_value = [1] * 100
        llm = LiteLLM(model="unknown-model")
        max_tokens = llm._get_max_token_for_prompt("test prompt")
        assert max_tokens > 0  # Should return a valid token count
        mock_get_encoding.assert_called_once_with("cl100k_base")


####################################
## Helper functions  and fixtures ##
####################################

add_tool = FunctionTool.from_defaults(fn=add, name="add")


def mock_chat_response(respx_mock: respx.MockRouter):
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Hello, world!"}}]},
        )
    )


def mock_completion_response(respx_mock: respx.MockRouter):
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": "Paris is the capital of France."}}]
            },
        )
    )


def mock_tool_response(respx_mock: respx.MockRouter):
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Let me calculate that for you.",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "add",
                                        "arguments": '{"x": 1, "y": 1}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )
    )


@pytest.fixture(autouse=True)
def setup_openai_api_key(monkeypatch):
    """Fixture to set up and tear down OPENAI_API_KEY for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture()
def llm():
    return LiteLLM(model="openai/gpt-fake-model")


def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny and 22 degrees {unit}."


def test_stream_tool_calls(respx_mock: respx.MockRouter, llm: LiteLLM):
    """Test streaming with tool calls being built up incrementally using respx to mock the network."""
    # Create the weather tool
    weather_tool = FunctionTool.from_defaults(fn=get_weather)

    # Create a sequence of streaming responses with properly escaped JSON
    stream_chunks = [
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"location\\""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"San Francisco, CA"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    # Set up the respx mock
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            content=b"".join(stream_chunks),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Call stream_chat and collect the results
    message = ChatMessage(role="user", content="What's the weather in San Francisco?")
    responses = list(
        llm.stream_chat([message], tools=[weather_tool.metadata.to_openai_tool()])
    )

    # Verify the results
    assert len(responses) > 0

    # Verify the final tool call is complete and correctly structured
    final_response = responses[-1]
    tool_calls = final_response.message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert tool_call["id"] == "call_abc123"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    # Check that the arguments were built up correctly
    expected_args = '{"location":"San Francisco, CA"}'
    assert tool_call["function"]["arguments"] == expected_args

    # Parse the arguments and verify they're valid JSON
    args = json.loads(tool_call["function"]["arguments"])
    assert args["location"] == "San Francisco, CA"


@pytest.mark.asyncio()
async def test_astream_tool_calls(respx_mock: respx.MockRouter, llm: LiteLLM):
    """Test async streaming with tool calls being built up incrementally."""
    # Create the weather tool
    weather_tool = FunctionTool.from_defaults(fn=get_weather)

    # Reuse the same stream chunks from the synchronous test
    stream_chunks = [
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"location\\""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"San Francisco, CA"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\""}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1716644530,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    # Set up the respx mock
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            content=b"".join(stream_chunks),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Call astream_chat and collect the results
    message = ChatMessage(role="user", content="What's the weather in San Francisco?")
    responses = []

    # Get the async stream and collect responses
    stream = await llm.astream_chat(
        [message], tools=[weather_tool.metadata.to_openai_tool()]
    )

    async for response in stream:
        responses.append(response)

    # Verify the results
    assert len(responses) > 0

    # Verify the final tool call is complete and correctly structured
    final_response = responses[-1]
    tool_calls = final_response.message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert tool_call["id"] == "call_abc123"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    # Check that the arguments were built up correctly
    expected_args = '{"location":"San Francisco, CA"}'
    assert tool_call["function"]["arguments"] == expected_args

    # Parse the arguments and verify they're valid JSON
    args = json.loads(tool_call["function"]["arguments"])
    assert args["location"] == "San Francisco, CA"


def test_image_block_chat(respx_mock: respx.MockRouter, llm: LiteLLM):
    """Test sending image blocks to OpenAI via LiteLLM."""
    # Mock the API response for a request with image blocks
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "I see a beautiful Golden Gate Bridge in San Francisco."
                        }
                    }
                ]
            },
        )
    )

    # Create a message with both image and text blocks
    image_url = "https://example.com/golden_gate.jpg"
    image_block = ImageBlock(url=image_url)
    text_block = TextBlock(text="What's in this image?")

    message = ChatMessage(role=MessageRole.USER, content=[image_block, text_block])

    # Send the message
    chat_response = llm.chat([message])

    # Check the response content
    assert (
        chat_response.message.blocks[0].text
        == "I see a beautiful Golden Gate Bridge in San Francisco."
    )

    # Verify the request was sent correctly (check the last request)
    request = respx_mock.calls.last.request
    request_json = json.loads(request.content)

    # Verify image block was correctly formatted in the request
    assert len(request_json["messages"]) == 1
    assert isinstance(request_json["messages"][0]["content"], list)

    # Check for both image and text blocks in the request
    content_blocks = request_json["messages"][0]["content"]
    assert len(content_blocks) == 2

    # Verify image block
    image_content = next(
        (item for item in content_blocks if item["type"] == "image_url"), None
    )
    assert image_content is not None
    assert image_content["image_url"]["url"] == image_url

    # Verify text block
    text_content = next(
        (item for item in content_blocks if item["type"] == "text"), None
    )
    assert text_content is not None
    assert text_content["text"] == "What's in this image?"
