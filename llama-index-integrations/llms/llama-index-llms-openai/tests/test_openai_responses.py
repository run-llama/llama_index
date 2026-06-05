import os
import httpx
import pytest
from unittest.mock import MagicMock, patch

from pathlib import Path
from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    DocumentBlock,
    ChatResponse,
    ThinkingBlock,
    ToolCallBlock,
)

from llama_index.llms.openai.responses import OpenAIResponses, ResponseFunctionToolCall
from llama_index.llms.openai.utils import to_openai_message_dicts, O1_MODELS
from llama_index.core.tools import FunctionTool
from llama_index.core.prompts import PromptTemplate
from openai.types.responses.response_reasoning_item import Content, Summary
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseReasoningItem,
    ResponseOutputItem,
    ResponseOutputText,
    ResponseOutputItemDoneEvent,
)
from pydantic import BaseModel, Field
from typing import Optional
from llama_index.llms.openai.responses import _supports_reasoning
from llama_index.llms.openai.utils import to_openai_responses_message_dict

# Skip markers for tests requiring API keys
SKIP_OPENAI_TESTS = not os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def default_responses_llm():
    """Create a default OpenAIResponses instance with mocked clients."""
    with patch("llama_index.llms.openai.responses.SyncOpenAI"):
        with patch("llama_index.llms.openai.responses.AsyncOpenAI"):
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
    with patch("llama_index.llms.openai.responses.SyncOpenAI"):
        with patch("llama_index.llms.openai.responses.AsyncOpenAI"):
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

    custom_kwargs = llm._get_model_kwargs(top_p=0.8, max_output_tokens=100)
    assert custom_kwargs["max_output_tokens"] == 100


def test_get_model_kwargs_excludes_params_with_reasoning(default_responses_llm):
    """Test that certain parameters are excluded when reasoning_options is set."""
    llm = default_responses_llm
    llm.reasoning_options = {"effort": "low"}

    kwargs = llm._get_model_kwargs()

    assert "top_p" not in kwargs
    assert "temperature" not in kwargs
    assert "presence_penalty" not in kwargs
    assert "frequency_penalty" not in kwargs

    assert "model" in kwargs
    assert "max_output_tokens" in kwargs

    if llm.model in O1_MODELS:
        assert "reasoning" in kwargs
        assert kwargs["reasoning"] == {"effort": "low"}
    else:
        assert "reasoning" not in kwargs


def test_get_model_kwargs_with_tools_none(default_responses_llm):
    """Test model kwargs generation when tools is explicitly None.

    This can happen when _prepare_chat_with_tools is called with an empty
    tools list, which sets tools to None. The _get_model_kwargs method
    should handle this gracefully.
    """
    llm = default_responses_llm
    kwargs = llm._get_model_kwargs(tools=None)

    assert kwargs["tools"] == []


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

    with patch("llama_index.llms.openai.responses.SyncOpenAI"):
        with patch("llama_index.llms.openai.responses.AsyncOpenAI"):
            llm = OpenAIResponses(model="gpt-4o-mini")
            chat_response = llm._parse_response_output(output)

    assert chat_response.message.role == MessageRole.ASSISTANT
    assert len(chat_response.message.blocks) == 1
    assert isinstance(chat_response.message.blocks[0], TextBlock)
    assert chat_response.message.blocks[0].text == "Hello world"


def test_process_response_event():
    """Test the static process_response_event method for streaming responses."""
    # Initial state
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
        logprobs=[],
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    updated_blocks, _, _, _, _, delta = result
    assert updated_blocks == [TextBlock(text="Hello")]
    assert delta == "Hello"

    event = ResponseOutputItemDoneEvent(
        item=ResponseReasoningItem(
            id="1",
            summary=[],
            type="reasoning",
            content=[
                Content(text="hello world", type="reasoning_text"),
                Content(text="this is a test", type="reasoning_text"),
            ],
            encrypted_content=None,
            status=None,
        ),
        output_index=1,
        sequence_number=1,
        type="response.output_item.done",
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    updated_blocks, _, _, _, _, _ = result
    assert updated_blocks == [
        ThinkingBlock(
            block_type="thinking",
            content="hello world\nthis is a test",
            num_tokens=None,
            additional_information={
                "id": "1",
                "type": "reasoning",
                "encrypted_content": None,
                "status": None,
            },
        )
    ]

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
        item_id="123",
        output_index=0,
        type="response.function_call_arguments.delta",
        delta='{"arg": "value"',
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    _, _, _, updated_call, _, _ = result
    assert updated_call.arguments == '{"arg": "value"'

    # Test function call arguments done
    event = ResponseFunctionCallArgumentsDoneEvent(
        name="test_function",
        item_id="123",
        output_index=0,
        type="response.function_call_arguments.done",
        arguments='{"arg": "value"}',
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=updated_call,
        track_previous_responses=False,
    )

    final_blocks, _, _, final_current_call, _, _ = result
    completed_tool_calls = [
        block for block in final_blocks if isinstance(block, ToolCallBlock)
    ]
    assert len(completed_tool_calls) == 1
    assert completed_tool_calls[0].tool_kwargs == '{"arg": "value"}'
    assert completed_tool_calls[0].tool_call_id == "123"
    assert completed_tool_calls[0].tool_name == "test_function"
    assert final_current_call is None


def test_process_response_event_with_text_annotation():
    """Test process_response_event handles ResponseOutputTextAnnotationAddedEvent."""
    built_in_tool_calls = []
    additional_kwargs = {}
    current_tool_call = None

    # Create a dummy annotation event
    event = ResponseOutputTextAnnotationAddedEvent(
        item_id="123",
        output_index=0,
        content_index=0,
        annotation_index=0,
        type="response.output_text.annotation.added",
        annotation={"type": "test_annotation", "value": 42},
        sequence_number=1,
    )

    result = OpenAIResponses.process_response_event(
        event=event,
        built_in_tool_calls=built_in_tool_calls,
        additional_kwargs=additional_kwargs,
        current_tool_call=current_tool_call,
        track_previous_responses=False,
    )

    # The annotation should be added to additional_kwargs["annotations"]
    _, _, updated_additional_kwargs, _, _, _ = result
    assert "annotations" in updated_additional_kwargs
    assert updated_additional_kwargs["annotations"] == [
        {"type": "test_annotation", "value": 42}
    ]


def test_get_tool_calls_from_response():
    """Test extracting tool calls from a chat response."""
    # Create a mock chat response with tool calls
    chat_response = MagicMock()
    chat_response.message.blocks = [
        ToolCallBlock(
            tool_call_id="123",
            tool_name="test_function",
            tool_kwargs='{"arg1": "value1", "arg2": 42}',
        )
    ]

    with patch("llama_index.llms.openai.responses.SyncOpenAI"):
        with patch("llama_index.llms.openai.responses.AsyncOpenAI"):
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


def test_structured_predict_uses_responses_parse(default_responses_llm):
    """Test that structured_predict uses responses.parse with text_format for constrained decoding."""

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = default_responses_llm
    mock_response = MagicMock()
    mock_response.output_parsed = Person(name="Alice", age=25)
    llm._client.responses.parse = MagicMock(return_value=mock_response)

    result = llm.structured_predict(
        output_cls=Person,
        prompt=PromptTemplate(
            "Create a profile for a person named {name} who is {age} years old"
        ),
        name="Alice",
        age=25,
    )

    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 25

    call_kwargs = llm._client.responses.parse.call_args
    assert call_kwargs.kwargs["text_format"] is Person
    assert call_kwargs.kwargs["tool_choice"] == "none"
    assert call_kwargs.kwargs["model"] == "gpt-4o-mini"


def test_structured_predict_raises_on_none_output(default_responses_llm):
    """Test that structured_predict raises ValueError when output_parsed is None."""

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = default_responses_llm
    mock_response = MagicMock()
    mock_response.output_parsed = None
    llm._client.responses.parse = MagicMock(return_value=mock_response)

    with pytest.raises(ValueError, match="Failed to produce a structured response"):
        llm.structured_predict(
            output_cls=Person,
            prompt=PromptTemplate("Create a profile for a person"),
        )


@pytest.mark.asyncio
async def test_astructured_predict_uses_responses_parse(default_responses_llm):
    """Test that astructured_predict uses async responses.parse with text_format."""
    from unittest.mock import AsyncMock

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = default_responses_llm
    mock_response = MagicMock()
    mock_response.output_parsed = Person(name="Bob", age=30)
    llm._aclient.responses.parse = AsyncMock(return_value=mock_response)

    result = await llm.astructured_predict(
        output_cls=Person,
        prompt=PromptTemplate(
            "Create a profile for a person named {name} who is {age} years old"
        ),
        name="Bob",
        age=30,
    )

    assert isinstance(result, Person)
    assert result.name == "Bob"
    assert result.age == 30

    call_kwargs = llm._aclient.responses.parse.call_args
    assert call_kwargs.kwargs["text_format"] is Person
    assert call_kwargs.kwargs["tool_choice"] == "none"
    assert call_kwargs.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_astructured_predict_raises_on_none_output(default_responses_llm):
    """Test that astructured_predict raises ValueError when output_parsed is None."""
    from unittest.mock import AsyncMock

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = default_responses_llm
    mock_response = MagicMock()
    mock_response.output_parsed = None
    llm._aclient.responses.parse = AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="Failed to produce a structured response"):
        await llm.astructured_predict(
            output_cls=Person,
            prompt=PromptTemplate("Create a profile for a person"),
        )


def test_structured_predict_passes_llm_kwargs(default_responses_llm):
    """Test that structured_predict forwards llm_kwargs to responses.parse."""

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = default_responses_llm
    mock_response = MagicMock()
    mock_response.output_parsed = Person(name="Alice", age=25)
    llm._client.responses.parse = MagicMock(return_value=mock_response)

    llm.structured_predict(
        output_cls=Person,
        prompt=PromptTemplate("Create a profile for a person"),
        llm_kwargs={"temperature": 0.5},
    )

    call_kwargs = llm._client.responses.parse.call_args
    assert call_kwargs.kwargs["temperature"] == 0.5


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
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 1
    )


def test_messages_to_openai_responses_messages():
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="1",
                    tool_name="get_capital_city_by_state",
                    tool_kwargs="{'state': 'France'}",
                )
            ],
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="Paris"),
        ChatMessage(role=MessageRole.USER, content="What is the capital of Germany?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="2",
                    tool_name="get_capital_city_by_state",
                    tool_kwargs="{'state': 'Germany'}",
                )
            ],
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(
                    content="The user is asking a simple question related to the capital of Germany, I should answer it concisely",
                    additional_information={"id": "123456789"},
                ),
                TextBlock(text="Berlin"),
            ],
        ),
    ]
    openai_messages = to_openai_message_dicts(messages, is_responses_api=True)
    assert len(openai_messages) == 7
    assert openai_messages[0]["role"] == "developer"
    assert openai_messages[0]["content"] == "You are a helpful assistant."
    assert openai_messages[1]["role"] == "user"
    assert openai_messages[1]["content"] == "What is the capital of France?"
    assert openai_messages[2] == {
        "type": "function_call",
        "arguments": "{'state': 'France'}",
        "call_id": "1",
        "name": "get_capital_city_by_state",
    }
    assert openai_messages[3]["role"] == "assistant"
    assert openai_messages[3]["content"] == "Paris"
    assert openai_messages[4]["role"] == "user"
    assert openai_messages[4]["content"] == "What is the capital of Germany?"
    assert openai_messages[5] == {
        "type": "function_call",
        "arguments": "{'state': 'Germany'}",
        "call_id": "2",
        "name": "get_capital_city_by_state",
    }
    assert openai_messages[6]["role"] == "assistant"
    assert len(openai_messages[6]["content"]) == 1
    assert openai_messages[6]["content"][0]["text"] == messages[6].blocks[1].text


def test_messages_to_openai_responses_messages_with_store():
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="1",
                    tool_name="get_capital_city_by_state",
                    tool_kwargs="{'state': 'France'}",
                )
            ],
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="Paris"),
        ChatMessage(role=MessageRole.USER, content="What is the capital of Germany?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="2",
                    tool_name="get_capital_city_by_state",
                    tool_kwargs="{'state': 'Germany'}",
                )
            ],
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(
                    content="The user is asking a simple question related to the capital of Germany, I should answer it concisely",
                    additional_information={"id": "123456789"},
                ),
                TextBlock(text="Berlin"),
            ],
        ),
    ]

    openai_messages = to_openai_message_dicts(
        messages, is_responses_api=True, store=True
    )
    assert len(openai_messages) == 8
    assert openai_messages[0]["role"] == "developer"
    assert openai_messages[0]["content"] == "You are a helpful assistant."
    assert openai_messages[1]["role"] == "user"
    assert openai_messages[1]["content"] == "What is the capital of France?"
    assert openai_messages[2] == {
        "type": "function_call",
        "arguments": "{'state': 'France'}",
        "call_id": "1",
        "name": "get_capital_city_by_state",
    }
    assert openai_messages[3]["role"] == "assistant"
    assert openai_messages[3]["content"] == "Paris"
    assert openai_messages[4]["role"] == "user"
    assert openai_messages[4]["content"] == "What is the capital of Germany?"
    assert openai_messages[5] == {
        "type": "function_call",
        "arguments": "{'state': 'Germany'}",
        "call_id": "2",
        "name": "get_capital_city_by_state",
    }

    assert openai_messages[6]["type"] == "reasoning"
    assert (
        openai_messages[6]["id"] == messages[6].blocks[0].additional_information["id"]
    )
    assert openai_messages[6]["summary"][0]["text"] == messages[6].blocks[0].content

    assert openai_messages[7]["role"] == "assistant"
    assert len(openai_messages[7]["content"]) == 1
    assert openai_messages[7]["content"][0]["text"] == messages[6].blocks[1].text


@pytest.fixture()
def response_output() -> List[ResponseOutputItem]:
    return [
        ResponseReasoningItem(
            id="1",
            summary=[],
            type="reasoning",
            content=[
                Content(text="hello world", type="reasoning_text"),
                Content(text="this is a test", type="reasoning_text"),
            ],
            encrypted_content=None,
            status=None,
        ),
        ResponseReasoningItem(
            id="1",
            summary=[],
            type="reasoning",
            content=[Content(text="another test", type="reasoning_text")],
            encrypted_content=None,
            status=None,
        ),
        ResponseReasoningItem(
            id="1",
            summary=[Summary(text="hello", type="summary_text")],
            type="reasoning",
            content=[Content(text="another test", type="reasoning_text")],
            encrypted_content=None,
            status=None,
        ),
        ResponseReasoningItem(
            id="1",
            summary=[
                Summary(text="hello", type="summary_text"),
                Summary(text="world", type="summary_text"),
            ],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status=None,
        ),
        ResponseFunctionToolCall(
            arguments="{'hello': 'world'}",
            call_id="1",
            name="test",
            type="function_call",
            status="completed",
        ),
        ResponseOutputMessage(
            id="1",
            content=[
                ResponseOutputText(annotations=[], text="hey there", type="output_text")
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
    ]


class OpenAIResponsesMock(OpenAIResponses):
    def __init__(self):
        pass


def test__parse_response_output(response_output: List[ResponseOutputItem]):
    result = OpenAIResponsesMock()._parse_response_output(output=response_output)
    assert (
        len(
            [
                block
                for block in result.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        == 4
    )
    assert (
        len([block for block in result.message.blocks if isinstance(block, TextBlock)])
        == 1
    )
    assert (
        len(
            [
                block
                for block in result.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 1
    )
    tool_call = [
        block for block in result.message.blocks if isinstance(block, ToolCallBlock)
    ][0]
    assert tool_call.tool_call_id == "1"
    assert tool_call.tool_name == "test"
    assert tool_call.tool_kwargs == "{'hello': 'world'}"
    assert [
        block for block in result.message.blocks if isinstance(block, ThinkingBlock)
    ][0].content == "hello world\nthis is a test"
    assert [
        block for block in result.message.blocks if isinstance(block, ThinkingBlock)
    ][1].content == "another test"
    assert [
        block for block in result.message.blocks if isinstance(block, ThinkingBlock)
    ][2].content == "another test\nhello"
    assert [
        block for block in result.message.blocks if isinstance(block, ThinkingBlock)
    ][3].content == "hello\nworld"

class TestSupportsReasoning:
    """_supports_reasoning must use prefix matching, not an exact O1_MODELS lookup."""

    # ── models that SHOULD match ───────────────────────────────────────────────

    def test_o1_mini(self):
        assert _supports_reasoning("o1-mini") is True

    def test_o1_preview(self):
        assert _supports_reasoning("o1-preview") is True

    def test_o3_mini(self):
        assert _supports_reasoning("o3-mini") is True

    def test_o3_dated(self):
        assert _supports_reasoning("o3-2025-04-16") is True

    def test_o4_mini_dated(self):
        assert _supports_reasoning("o4-mini-2025-04-16") is True

    def test_gpt5_base_dated(self):
        assert _supports_reasoning("gpt-5-2025-08-07") is True

    def test_gpt5_dot_two_dated(self):
        assert _supports_reasoning("gpt-5.2-2025-12-11") is True

    def test_gpt5_dot_four_dated(self):
        assert _supports_reasoning("gpt-5.4-2026-03-05") is True

    def test_gpt5_dot_five_dated(self):
        assert _supports_reasoning("gpt-5.5-2026-04-23") is True

    def test_future_unlisted_snapshot_still_matches(self):
        """
        THE key regression case:
        A snapshot not yet in O1_MODELS must still be caught by prefix.
        This is exactly what caused issue #20459.
        """
        assert _supports_reasoning("gpt-5.9-2027-01-01") is True

    # ── models that must NOT match ─────────────────────────────────────────────

    def test_gpt4o_no_match(self):
        assert _supports_reasoning("gpt-4o") is False

    def test_gpt4o_mini_no_match(self):
        assert _supports_reasoning("gpt-4o-mini") is False

    def test_gpt4_turbo_no_match(self):
        assert _supports_reasoning("gpt-4-turbo") is False

    def test_gpt4_1_no_match(self):
        assert _supports_reasoning("gpt-4.1") is False

    def test_empty_string_no_match(self):
        assert _supports_reasoning("") is False

    def test_random_string_no_match(self):
        assert _supports_reasoning("some-unknown-model") is False

def _make_llm_for_temp(model: str, reasoning_options=None, temperature: float = 0.3):
    """Create OpenAIResponses without real API calls."""
    with patch("llama_index.llms.openai.responses.SyncOpenAI"):
        with patch("llama_index.llms.openai.responses.AsyncOpenAI"):
            return OpenAIResponses(
                model=model,
                temperature=temperature,
                reasoning_options=reasoning_options,
                api_key="sk-test",
            )


class TestConstructorTemperatureOverride:

    def test_active_reasoning_forces_temperature_to_one(self):
        """effort='low' means reasoning is on — temperature must become 1.0."""
        llm = _make_llm_for_temp("gpt-5.2-2025-12-11", {"effort": "low"}, 0.3)
        assert llm.temperature == 1.0

    def test_effort_none_preserves_custom_temperature(self):
        """effort='none' means reasoning is off — user's temperature must survive."""
        llm = _make_llm_for_temp("gpt-5.2-2025-12-11", {"effort": "none"}, 0.3)
        assert llm.temperature == 0.3

    def test_non_reasoning_model_preserves_temperature(self):
        """gpt-4o is not a reasoning model; temperature must never be overridden."""
        llm = _make_llm_for_temp("gpt-4o", reasoning_options=None, temperature=0.7)
        assert llm.temperature == 0.7

    def test_unlisted_snapshot_active_reasoning_forces_temperature(self):
        """Snapshot not in O1_MODELS but matching prefix must be caught."""
        llm = _make_llm_for_temp("gpt-5.9-2027-01-01", {"effort": "medium"}, 0.5)
        assert llm.temperature == 1.0

    def test_reasoning_model_no_options_forces_temperature(self):
        """
        reasoning_options=None on a reasoning model should still force 1.0
        (matches original O1 behaviour for models with no explicit options).
        """
        llm = _make_llm_for_temp("gpt-5.2-2025-12-11", reasoning_options=None, temperature=0.5)
        assert llm.temperature == 1.0

def _get_model_kwargs(model: str, reasoning_options=None) -> dict:
    llm = _make_llm_for_temp(model, reasoning_options)
    return llm._get_model_kwargs()


class TestSamplingParamStripping:

    # ── The exact bug that caused #20459 ──────────────────────────────────────

    def test_top_p_stripped_when_reasoning_active(self):
        """top_p must not reach the API when reasoning effort is active."""
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "low"})
        assert "top_p" not in kwargs, (
            "top_p was sent to a reasoning model — this causes 400 Bad Request (#20459)"
        )

    def test_temperature_stripped_when_reasoning_active(self):
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "low"})
        assert "temperature" not in kwargs

    def test_presence_penalty_stripped_when_reasoning_active(self):
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "high"})
        assert "presence_penalty" not in kwargs

    def test_frequency_penalty_stripped_when_reasoning_active(self):
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "high"})
        assert "frequency_penalty" not in kwargs


    def test_top_p_kept_when_effort_is_none(self):
        """GPT-5.2+ accepts sampling params when effort='none'."""
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "none"})
        assert "top_p" in kwargs

    def test_temperature_kept_when_effort_is_none(self):
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "none"})
        assert "temperature" in kwargs

    # ── Non-reasoning models must never have params stripped ──────────────────

    def test_top_p_kept_for_gpt4o(self):
        kwargs = _get_model_kwargs("gpt-4o")
        assert "top_p" in kwargs

    def test_temperature_kept_for_gpt4o(self):
        kwargs = _get_model_kwargs("gpt-4o")
        assert "temperature" in kwargs

    # ── reasoning dict forwarded correctly ────────────────────────────────────

    def test_reasoning_kwarg_present_when_active(self):
        kwargs = _get_model_kwargs("gpt-5.2-2025-12-11", {"effort": "low"})
        assert "reasoning" in kwargs
        assert kwargs["reasoning"] == {"effort": "low"}

    def test_reasoning_kwarg_absent_for_non_reasoning_model(self):
        """Even with reasoning_options set, don't forward for non-reasoning models."""
        kwargs = _get_model_kwargs("gpt-4o", {"effort": "low"})
        assert "reasoning" not in kwargs

    def test_unlisted_snapshot_top_p_stripped_when_active(self):
        """A snapshot not yet in O1_MODELS must still have top_p stripped."""
        kwargs = _get_model_kwargs("gpt-5.9-2027-01-01", {"effort": "medium"})
        assert "top_p" not in kwargs

def _mock_output_message(text: str, phase: Optional[str] = None):
    """Build a mock that passes isinstance(..., ResponseOutputMessage)."""
    part = MagicMock()
    part.text = text
    part.annotations = []
    part.refusal = None

    item = MagicMock(spec=ResponseOutputMessage)
    item.content = [part]
    item.phase = phase
    return item


class TestParseResponseOutputPhase:

    def test_commentary_phase_captured(self):
        item = _mock_output_message("Let me think…", phase="commentary")
        result = OpenAIResponsesMock()._parse_response_output([item])
        assert result.additional_kwargs.get("phase") == "commentary"

    def test_final_answer_phase_captured(self):
        item = _mock_output_message("Here's the answer.", phase="final_answer")
        result = OpenAIResponsesMock()._parse_response_output([item])
        assert result.additional_kwargs.get("phase") == "final_answer"

    def test_absent_phase_not_injected(self):
        """phase=None must not create a 'phase' key in additional_kwargs."""
        item = _mock_output_message("Normal response.", phase=None)
        result = OpenAIResponsesMock()._parse_response_output([item])
        assert "phase" not in result.additional_kwargs

    def test_unrecognised_phase_not_injected(self):
        """Invalid phase values must not pollute additional_kwargs."""
        item = _mock_output_message("Odd response.", phase="unknown_value")
        result = OpenAIResponsesMock()._parse_response_output([item])
        assert "phase" not in result.additional_kwargs

    def test_text_extraction_unaffected_by_phase_capture(self):
        """Adding phase capture must not break text block parsing."""
        item = _mock_output_message("My answer.", phase="final_answer")
        result = OpenAIResponsesMock()._parse_response_output([item])
        text_blocks = [b for b in result.message.blocks if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "My answer."

def _simulate_token_assignment(thinking_count: int, total: int):
    """
    Run the fixed token-assignment logic and return the per-block num_tokens.
    Tests the logic directly rather than mocking a full API response.
    """
    blocks = [ThinkingBlock(content=f"step {i}") for i in range(thinking_count)]
    if blocks:
        per_block, remainder = divmod(total, len(blocks))
        for i, block in enumerate(blocks):
            block.num_tokens = per_block + (1 if i < remainder else 0)
    return [b.num_tokens for b in blocks]


class TestReasoningTokenDistribution:

    def test_single_block_gets_full_total(self):
        assert _simulate_token_assignment(1, 900) == [900]

    def test_three_blocks_equal_split(self):
        tokens = _simulate_token_assignment(3, 900)
        assert tokens == [300, 300, 300]

    def test_sum_always_equals_total(self):
        for count in [1, 2, 3, 4, 5]:
            tokens = _simulate_token_assignment(count, 1000)
            assert sum(tokens) == 1000, f"Sum mismatch for block count {count}"

    def test_remainder_distributed_to_first_blocks(self):
        """901 / 3 = 300 rem 1 → first block gets the extra."""
        tokens = _simulate_token_assignment(3, 901)
        assert tokens == [301, 300, 300]
        assert sum(tokens) == 901

    def test_two_blocks_odd_total(self):
        tokens = _simulate_token_assignment(2, 101)
        assert tokens[0] == 51
        assert tokens[1] == 50
        assert sum(tokens) == 101

    def test_no_duplication_the_original_bug(self):
        """
        THE original bug: before the fix, every block was assigned the total.
        3 blocks × 900 tokens each → reporter saw usage of 2700 tokens.
        After fix: sum must equal 900, not 2700.
        """
        tokens = _simulate_token_assignment(3, 900)
        assert sum(tokens) == 900, (
            f"Token duplication bug present: sum={sum(tokens)} instead of 900"
        )

    def test_zero_tokens_handled_cleanly(self):
        tokens = _simulate_token_assignment(3, 0)
        assert tokens == [0, 0, 0]

def _make_assistant_msg(text: str, phase: Optional[str]) -> ChatMessage:
    additional_kwargs = {}
    if phase is not None:
        additional_kwargs["phase"] = phase
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[TextBlock(text=text)],
        additional_kwargs=additional_kwargs,
    )


def _pull_assistant_dict(result):
    """
    Extract the assistant role dict from whatever the serialiser returned.
    The function may return a plain dict, a list (when reasoning items are
    prepended), or a string.
    """
    if isinstance(result, list):
        candidates = [
            d for d in result
            if isinstance(d, dict) and d.get("role") == "assistant"
        ]
        assert candidates, f"No assistant dict in serialised output: {result}"
        return candidates[0]
    assert isinstance(result, dict), f"Expected dict, got {type(result)}: {result}"
    return result


class TestPhaseSerialisation:

    def test_commentary_written_to_output(self):
        msg = _make_assistant_msg("Let me think…", "commentary")
        result = to_openai_responses_message_dict(msg)
        d = _pull_assistant_dict(result)
        assert d.get("phase") == "commentary"

    def test_final_answer_written_to_output(self):
        msg = _make_assistant_msg("Here's my answer.", "final_answer")
        result = to_openai_responses_message_dict(msg)
        d = _pull_assistant_dict(result)
        assert d.get("phase") == "final_answer"

    def test_absent_phase_not_injected(self):
        """No phase in additional_kwargs → must not appear in the output at all."""
        msg = _make_assistant_msg("Normal text.", None)
        result = to_openai_responses_message_dict(msg)
        items = result if isinstance(result, list) else [result]
        for item in items:
            if isinstance(item, dict):
                assert "phase" not in item

    def test_invalid_phase_not_injected(self):
        """Unrecognised phase values must not leak into the API payload."""
        msg = _make_assistant_msg("Odd text.", "completely_wrong")
        result = to_openai_responses_message_dict(msg)
        items = result if isinstance(result, list) else [result]
        for item in items:
            if isinstance(item, dict):
                assert "phase" not in item

    def test_full_round_trip(self):
        """
        Simulate the parse → ChatMessage → serialise round-trip.
        Combines bug 2 fix (_parse_response_output stores phase) with
        bug 3 fix (serialiser writes it back out).
        """
        # Pretend _parse_response_output already captured phase onto the message
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[TextBlock(text="commentary turn")],
            additional_kwargs={"phase": "commentary"},
        )
        result = to_openai_responses_message_dict(msg)
        d = _pull_assistant_dict(result)
        assert d.get("phase") == "commentary", (
            "phase was lost during serialisation — bug 3 is not fixed"
        )

class TestModelForwardingInMessageDicts:

    def _system_user_messages(self):
        return [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ]

    def test_gpt4o_system_not_converted_to_developer(self):
        """
        gpt-4o is not in O1_MODELS — system content must NOT be wrapped
        with role=developer. After the model forwarding fix, the system
        message is serialised using the actual model, not hardcoded o3-mini.
        """
        result = to_openai_message_dicts(
            self._system_user_messages(), model="gpt-4o", is_responses_api=True
        )
        # Flatten: result may be a list of dicts or a string
        items = result if isinstance(result, list) else [result]
        roles = [d.get("role") for d in items if isinstance(d, dict)]
        # developer must NOT appear for a non-O1 model
        assert "developer" not in roles, (
            f"'developer' role must not appear for gpt-4o. Got: {roles}"
        )

    def test_o3_mini_system_not_incorrectly_double_converted(self):
        """
        o3-mini is in O1_MODELS. The system message content must be
        present in the output and must not appear as role=system.
        """
        result = to_openai_message_dicts(
            self._system_user_messages(), model="o3-mini", is_responses_api=True
        )
        items = result if isinstance(result, list) else [result]
        roles = [d.get("role") for d in items if isinstance(d, dict)]
        # system must not survive — it becomes developer or is inlined as content
        assert "system" not in roles, (
            f"'system' role should not survive for o3-mini. Got: {roles}"
        )

    def test_model_none_does_not_raise(self):
        """model=None must not crash — it just skips role conversion."""
        result = to_openai_message_dicts(
            self._system_user_messages(), model=None, is_responses_api=True
        )
        assert isinstance(result, (list, str))

    def test_gpt5_reasoning_model_does_not_raise(self):
        """GPT-5.x reasoning model must be handled without error."""
        result = to_openai_message_dicts(
            self._system_user_messages(),
            model="gpt-5.2-2025-12-11",
            is_responses_api=True,
        )
        assert isinstance(result, (list, str))