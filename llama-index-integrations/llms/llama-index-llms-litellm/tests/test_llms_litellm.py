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
)


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
