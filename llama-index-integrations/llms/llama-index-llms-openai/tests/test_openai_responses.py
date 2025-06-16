import os
import httpx
import pytest
from unittest.mock import MagicMock, patch

from pathlib import Path
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    DocumentBlock,
    ChatResponse,
)
from llama_index.llms.openai.responses import OpenAIResponses, ResponseFunctionToolCall
from llama_index.llms.openai.utils import to_openai_message_dicts
from llama_index.core.tools import FunctionTool
from llama_index.core.prompts import PromptTemplate
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)
from pydantic import BaseModel, Field

# Skip markers for tests requiring API keys
SKIP_OPENAI_TESTS = not os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def default_responses_llm():
    """Create a default OpenAIResponses instance with mocked clients."""
    with (
        patch("llama_index.llms.openai.responses.SyncOpenAI"),
        patch("llama_index.llms.openai.responses.AsyncOpenAI"),
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
    with (
        patch("llama_index.llms.openai.responses.SyncOpenAI"),
        patch("llama_index.llms.openai.responses.AsyncOpenAI"),
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

    with (
        patch("llama_index.llms.openai.responses.SyncOpenAI"),
        patch("llama_index.llms.openai.responses.AsyncOpenAI"),
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
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        tool_calls=tool_calls,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    updated_blocks, updated_tool_calls, _, _, _, _, delta = result
    assert updated_blocks == [TextBlock(text="Hello")]
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
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
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
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
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


def test_process_response_event_with_text_annotation():
    """Test process_response_event handles ResponseOutputTextAnnotationAddedEvent."""
    tool_calls = []
    built_in_tool_calls = []
    additional_kwargs = {}
    current_tool_call = None

    # Create a dummy annotation event
    event = ResponseOutputTextAnnotationAddedEvent(
        item_id="123",
        output_index=0,
        content_index=0,
        annotation_index=0,
        type="response.output_text_annotation.added",
        annotation={"type": "test_annotation", "value": 42},
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        tool_calls=tool_calls,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    # The annotation should be added to additional_kwargs["annotations"]
    _, _, _, updated_additional_kwargs, _, _, _ = result
    assert "annotations" in updated_additional_kwargs
    assert updated_additional_kwargs["annotations"] == [
        {"type": "test_annotation", "value": 42}
    ]


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

    with (
        patch("llama_index.llms.openai.responses.SyncOpenAI"),
        patch("llama_index.llms.openai.responses.AsyncOpenAI"),
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


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required=True is correctly passed to the API request in OpenAIResponses."""
    # Create mock clients to avoid API calls
    mock_sync_client = MagicMock()
    mock_async_client = MagicMock()

    llm = OpenAIResponses(api_key="test-key")
    llm._client = mock_sync_client
    llm._aclient = mock_async_client

    # Create a simple tool for testing
    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    search_tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    # Test with tool_required=True
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

    assert result["tool_choice"] == "required"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required=False is correctly passed to the API request in OpenAIResponses."""
    # Create mock clients to avoid API calls
    mock_sync_client = MagicMock()
    mock_async_client = MagicMock()

    llm = OpenAIResponses(api_key="test-key")
    llm._client = mock_sync_client
    llm._aclient = mock_async_client

    # Create a simple tool for testing
    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    search_tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=False)

    assert result["tool_choice"] == "auto"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_tools_explicit_tool_choice_overrides_tool_required():
    """Test that explicit tool_choice overrides tool_required in OpenAIResponses."""
    # Create mock clients to avoid API calls
    mock_sync_client = MagicMock()
    mock_async_client = MagicMock()

    llm = OpenAIResponses(api_key="test-key")
    llm._client = mock_sync_client
    llm._aclient = mock_async_client

    # Create a simple tool for testing
    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    search_tool = FunctionTool.from_defaults(
        fn=search, name="search_tool", description="A tool for searching information"
    )

    # Test that explicit tool_choice overrides tool_required
    result = llm._prepare_chat_with_tools(
        tools=[search_tool], tool_required=True, tool_choice="none"
    )

    assert result["tool_choice"] == "none"  # Should be "none" not "required"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


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
    accumulated_content = "".join([r.delta for r in responses if r.delta is not None])
    assert len(accumulated_content) > 0, "Accumulated content should not be empty"


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
    accumulated_content = "".join([r.delta for r in responses if r.delta is not None])
    assert len(accumulated_content) > 0, "Accumulated content should not be empty"


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


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_document_upload(tmp_path: Path, pdf_url: str) -> None:
    llm = OpenAIResponses(model="gpt-4.1")
    pdf_path = tmp_path / "test.pdf"
    pdf_content = httpx.get(pdf_url).content
    pdf_path.write_bytes(pdf_content)
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            DocumentBlock(path=pdf_path),
            TextBlock(text="What does the document contain?"),
        ],
    )
    messages = [msg]
    response = llm.chat(messages)
    assert isinstance(response, ChatResponse)


def search(query: str) -> str:
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(fn=search)


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI API key not available")
def test_tool_required():
    llm = OpenAIResponses(model="gpt-4.1-mini")
    response = llm.chat_with_tools(
        user_msg="What is the capital of France?",
        tools=[search_tool],
        tool_required=True,
    )
    assert len(response.message.additional_kwargs["tool_calls"]) == 1


def test_messages_to_openai_responses_messages():
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Paris"),
        ChatMessage(role=MessageRole.USER, content="What is the capital of Germany?"),
    ]
    openai_messages = to_openai_message_dicts(messages, is_responses_api=True)
    assert len(openai_messages) == 4
    assert openai_messages[0]["role"] == "developer"
    assert openai_messages[0]["content"] == "You are a helpful assistant."
    assert openai_messages[1]["role"] == "user"
    assert openai_messages[1]["content"] == "What is the capital of France?"
    assert openai_messages[2]["role"] == "assistant"
    assert openai_messages[2]["content"] == "Paris"
    assert openai_messages[3]["role"] == "user"
    assert openai_messages[3]["content"] == "What is the capital of Germany?"
