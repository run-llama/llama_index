import os
import pytest
from unittest.mock import MagicMock, patch

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai.responses import OpenAIResponses, ResponseFunctionToolCall
from llama_index.core.tools import FunctionTool
from llama_index.core.prompts import PromptTemplate
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)
from pydantic import BaseModel, Field

# Skip markers for tests requiring API keys
SKIP_OPENAI_TESTS = not os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def default_responses_llm():
    """Create a default OpenAIResponses instance with mocked clients."""
    with patch("llama_index.llms.openai.responses.SyncOpenAI"), patch(
        "llama_index.llms.openai.responses.AsyncOpenAI"
    ):
        llm = OpenAIResponses(
            model="gpt-4o-mini",
            api_key="fake-api-key",
            api_base="https://api.openai.com/v1",
            api_version="2023-05-15",
        )
    return llm


def test_init_and_properties(default_responses_llm):
    """Test initialization and property access."""
    llm = default_responses_llm

    assert llm.model == "gpt-4o-mini"
    assert llm.temperature == 0.1
    assert llm.max_retries == 3

    metadata = llm.metadata
    assert metadata.model_name == "gpt-4o-mini"
    assert metadata.is_chat_model is True


def test_get_model_name():
    """Test different model name formats are properly handled."""
    with patch("llama_index.llms.openai.responses.SyncOpenAI"), patch(
        "llama_index.llms.openai.responses.AsyncOpenAI"
    ):
        # Standard model
        llm = OpenAIResponses(model="gpt-4o-mini")
        assert llm._get_model_name() == "gpt-4o-mini"

        # Legacy fine-tuning format
        llm = OpenAIResponses(model="ft-model:gpt-4")
        assert llm._get_model_name() == "ft-model"

        # New fine-tuning format
        llm = OpenAIResponses(model="ft:gpt-4:org:custom:id")
        assert llm._get_model_name() == "gpt-4"


def test_get_model_kwargs(default_responses_llm):
    """Test model kwargs generation."""
    llm = default_responses_llm
    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.1
    assert kwargs["truncation"] == "disabled"

    # Test with additional kwargs
    custom_kwargs = llm._get_model_kwargs(top_p=0.8, max_output_tokens=100)
    assert custom_kwargs["top_p"] == 0.8
    assert custom_kwargs["max_output_tokens"] == 100


def test_parse_response_output():
    """Test parsing response output into ChatResponse."""
    # Create mock output items
    output = [
        ResponseOutputMessage(
            type="message",
            content=[{"type": "output_text", "text": "Hello world", "annotations": []}],
            role="assistant",
            id="123",
            status="completed",
        )
    ]

    with patch("llama_index.llms.openai.responses.SyncOpenAI"), patch(
        "llama_index.llms.openai.responses.AsyncOpenAI"
    ):
        llm = OpenAIResponses(model="gpt-4o-mini")
        chat_response = llm._parse_response_output(output)

    assert chat_response.message.role == MessageRole.ASSISTANT
    assert len(chat_response.message.blocks) == 1
    assert isinstance(chat_response.message.blocks[0], TextBlock)
    assert chat_response.message.blocks[0].text == "Hello world"


def test_process_response_event():
    """Test the static process_response_event method for streaming responses."""
    # Initial state
    content = ""
    tool_calls = []
    built_in_tool_calls = []
    additional_kwargs = {}
    current_tool_call = None

    # Test text delta event
    event = ResponseTextDeltaEvent(
        content_index=0,
        item_id="123",
        output_index=0,
        delta="Hello",
        type="response.output_text.delta",
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        content=content,
        tool_calls=tool_calls,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    updated_content, updated_tool_calls, _, _, _, _, delta = result
    assert updated_content == "Hello"
    assert delta == "Hello"
    assert updated_tool_calls == []

    # Test function call arguments delta
    current_tool_call = ResponseFunctionToolCall(
        id="call_123",
        call_id="123",
        type="function_call",
        name="test_function",
        arguments="",
        status="in_progress",
    )

    event = ResponseFunctionCallArgumentsDeltaEvent(
        content_index=0,
        item_id="123",
        output_index=0,
        type="response.function_call_arguments.delta",
        delta='{"arg": "value"',
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        content=updated_content,
        tool_calls=updated_tool_calls,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    _, _, _, _, updated_call, _, _ = result
    assert updated_call.arguments == '{"arg": "value"'

    # Test function call arguments done
    event = ResponseFunctionCallArgumentsDoneEvent(
        item_id="123",
        output_index=0,
        type="response.function_call_arguments.done",
        arguments='{"arg": "value"}',
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        content=updated_content,
        tool_calls=updated_tool_calls,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=updated_call,
        track_previous_responses=False,
    )

    _, completed_tool_calls, _, _, final_current_call, _, _ = result
    assert len(completed_tool_calls) == 1
    assert completed_tool_calls[0].arguments == '{"arg": "value"}'
    assert completed_tool_calls[0].status == "completed"
    assert final_current_call is None


def test_get_tool_calls_from_response():
    """Test extracting tool calls from a chat response."""
    tool_call = ResponseFunctionToolCall(
        id="call_123",
        call_id="123",
        type="function_call",
        name="test_function",
        arguments='{"arg1": "value1", "arg2": 42}',
        status="completed",
    )

    # Create a mock chat response with tool calls
    chat_response = MagicMock()
    chat_response.message.additional_kwargs = {"tool_calls": [tool_call]}

    with patch("llama_index.llms.openai.responses.SyncOpenAI"), patch(
        "llama_index.llms.openai.responses.AsyncOpenAI"
    ):
        llm = OpenAIResponses(model="gpt-4o-mini")
        tool_selections = llm.get_tool_calls_from_response(chat_response)

    assert len(tool_selections) == 1
    assert tool_selections[0].tool_id == "123"
    assert tool_selections[0].tool_name == "test_function"
    assert tool_selections[0].tool_kwargs == {"arg1": "value1", "arg2": 42}


def test_prepare_chat_with_tools(default_responses_llm):
    """Test preparing a chat with tools."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = FunctionTool.from_defaults(fn=add)

    result = default_responses_llm._prepare_chat_with_tools(
        tools=[tool],
        user_msg="What is 2+2?",
        allow_parallel_tool_calls=False,
    )

    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["name"] == "add"
    assert result["parallel_tool_calls"] is False

    # Check that the message was properly formatted
    assert len(result["messages"]) == 1
    assert result["messages"][0].role == MessageRole.USER
    assert result["messages"][0].content == "What is 2+2?"


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_chat_with_api():
    """Test the chat method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")
    messages = [ChatMessage(role=MessageRole.USER, content="Say hello")]

    response = llm.chat(messages)
    assert response.message.role == MessageRole.ASSISTANT
    assert len(response.message.blocks) > 0


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_complete_with_api():
    """Test the complete method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")

    response = llm.complete("Write a one-sentence summary of Python.")
    assert response.text is not None
    assert len(response.text) > 0


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_stream_chat_with_api():
    """Test the stream_chat method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")
    messages = [ChatMessage(role=MessageRole.USER, content="Count to 3")]

    response_gen = llm.stream_chat(messages)
    responses = list(response_gen)

    assert len(responses) > 0
    assert all(r.message.role == MessageRole.ASSISTANT for r in responses)
    assert responses[-1].message.content is not None


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_stream_complete_with_api():
    """Test the stream_complete method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")

    response_gen = llm.stream_complete("Count to 3 briefly.")
    responses = list(response_gen)

    assert len(responses) > 0
    assert responses[-1].text is not None


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
@pytest.mark.asyncio
async def test_achat_with_api():
    """Test the async chat method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")
    messages = [ChatMessage(role=MessageRole.USER, content="Say hello")]

    response = await llm.achat(messages)
    assert response.message.role == MessageRole.ASSISTANT
    assert len(response.message.blocks) > 0


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
@pytest.mark.asyncio
async def test_acomplete_with_api():
    """Test the async complete method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")

    response = await llm.acomplete("Write a one-sentence summary of Python.")
    assert response.text is not None
    assert len(response.text) > 0


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
@pytest.mark.asyncio
async def test_astream_chat_with_api():
    """Test the async streaming chat method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")
    messages = [ChatMessage(role=MessageRole.USER, content="Count to 3")]

    response_gen = await llm.astream_chat(messages)
    responses = [resp async for resp in response_gen]

    assert len(responses) > 0
    assert all(r.message.role == MessageRole.ASSISTANT for r in responses)
    assert responses[-1].message.content is not None


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
@pytest.mark.asyncio
async def test_astream_complete_with_api():
    """Test the async stream_complete method with real API call."""
    llm = OpenAIResponses(model="gpt-4o-mini")

    response_gen = await llm.astream_complete("Count to 3 briefly.")
    responses = [resp async for resp in response_gen]

    assert len(responses) > 0
    assert responses[-1].text is not None


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_structured_prediction_with_api():
    """Test structured prediction with real API call."""

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = OpenAIResponses(model="gpt-4o-mini")
    result = llm.structured_predict(
        output_cls=Person,
        prompt=PromptTemplate(
            "Create a profile for a person named Alice who is 25 years old"
        ),
    )

    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 25


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_chat_with_built_in_tools():
    """Test chat with built-in tools enabled."""
    llm = OpenAIResponses(model="gpt-4o-mini", built_in_tools=[{"type": "web_search"}])

    messages = [
        ChatMessage(
            role=MessageRole.USER, content="What is the current time in New York City?"
        )
    ]

    response = llm.chat(messages)

    # We can't assert exactly what will be returned, but we can check structure
    assert response.message.role == MessageRole.ASSISTANT
    assert len(response.message.blocks) > 0

    # Should contain built-in tool calls in the response
    assert "built_in_tool_calls" in response.additional_kwargs
