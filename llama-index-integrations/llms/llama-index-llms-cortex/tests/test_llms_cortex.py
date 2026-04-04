from unittest.mock import MagicMock

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms import ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.llms.cortex import Cortex
from llama_index.llms.cortex.base import TOOL_CALLING_MODELS


def _make_cortex(**overrides):
    """Create a Cortex instance bypassing auth validation via model_construct."""
    defaults = {
        "model": "llama3.2-1b",
        "user": "test_user",
        "account": "test-account",
        "private_key_file": None,
        "jwt_token": "fake-token",
        "session": None,
        "context_window": 128_000,
        "max_tokens": 4096,
    }
    defaults.update(overrides)
    return Cortex.model_construct(**defaults)


def test_base_class_hierarchy():
    """Cortex should extend FunctionCallingLLM (and transitively BaseLLM)."""
    names_of_base_classes = [b.__name__ for b in Cortex.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
    assert FunctionCallingLLM.__name__ in names_of_base_classes


def test_metadata_function_calling_enabled():
    """Tool-calling models should report is_function_calling_model=True."""
    for model_name in TOOL_CALLING_MODELS:
        llm = _make_cortex(model=model_name, context_window=200_000)
        assert llm.metadata.is_function_calling_model is True


def test_metadata_function_calling_disabled():
    """Non-tool-calling models should report is_function_calling_model=False."""
    llm = _make_cortex(model="llama3.2-1b")
    assert llm.metadata.is_function_calling_model is False


def test_prepare_chat_with_tools():
    """_prepare_chat_with_tools should convert tools to Snowflake tool_spec format."""
    llm = _make_cortex(model="claude-3-5-sonnet")

    mock_tool = MagicMock()
    mock_tool.metadata.name = "get_weather"
    mock_tool.metadata.description = "Get the current weather"
    mock_tool.metadata.get_parameters_dict.return_value = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    }

    result = llm._prepare_chat_with_tools(
        tools=[mock_tool],
        user_msg="What's the weather in SF?",
    )

    assert "messages" in result
    assert "tools" in result
    assert "tool_choice" in result

    assert len(result["messages"]) == 1
    assert result["messages"][0].role == MessageRole.USER

    tool_spec = result["tools"][0]
    assert "tool_spec" in tool_spec
    assert tool_spec["tool_spec"]["type"] == "generic"
    assert tool_spec["tool_spec"]["name"] == "get_weather"
    assert tool_spec["tool_spec"]["description"] == "Get the current weather"
    assert "input_schema" in tool_spec["tool_spec"]

    assert result["tool_choice"] == {"type": "auto"}


def test_prepare_chat_with_tools_required():
    """When tool_required=True, tool_choice should be 'any'."""
    llm = _make_cortex(model="claude-3-5-sonnet")

    mock_tool = MagicMock()
    mock_tool.metadata.name = "search"
    mock_tool.metadata.description = "Search"
    mock_tool.metadata.get_parameters_dict.return_value = {
        "type": "object",
        "properties": {},
    }

    result = llm._prepare_chat_with_tools(
        tools=[mock_tool],
        user_msg="Search for X",
        tool_required=True,
    )

    assert result["tool_choice"] == {"type": "any"}


def test_prepare_chat_with_tools_no_tools():
    """When no tools provided, tools list should be empty and no tool_choice."""
    llm = _make_cortex(model="claude-3-5-sonnet")

    result = llm._prepare_chat_with_tools(
        tools=[],
        user_msg="Hello",
    )

    assert result["tools"] == []
    assert "tool_choice" not in result


def test_get_tool_calls_from_response():
    """get_tool_calls_from_response should extract ToolSelection from ToolCallBlocks."""
    llm = _make_cortex()

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                TextBlock(text="Let me check the weather."),
                ToolCallBlock(
                    tool_call_id="call_123",
                    tool_name="get_weather",
                    tool_kwargs={"location": "San Francisco"},
                ),
            ],
        ),
    )

    selections = llm.get_tool_calls_from_response(response)

    assert len(selections) == 1
    assert isinstance(selections[0], ToolSelection)
    assert selections[0].tool_id == "call_123"
    assert selections[0].tool_name == "get_weather"
    assert selections[0].tool_kwargs == {"location": "San Francisco"}


def test_get_tool_calls_from_response_multiple():
    """Should handle multiple parallel tool calls."""
    llm = _make_cortex()

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="get_weather",
                    tool_kwargs={"location": "SF"},
                ),
                ToolCallBlock(
                    tool_call_id="call_2",
                    tool_name="get_weather",
                    tool_kwargs={"location": "NYC"},
                ),
            ],
        ),
    )

    selections = llm.get_tool_calls_from_response(response)
    assert len(selections) == 2
    assert selections[0].tool_name == "get_weather"
    assert selections[0].tool_kwargs == {"location": "SF"}
    assert selections[1].tool_kwargs == {"location": "NYC"}


def test_get_tool_calls_from_response_no_calls():
    """Should return empty list when error_on_no_tool_call=False."""
    llm = _make_cortex()

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Just a text response.",
        ),
    )

    selections = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    assert selections == []


def test_get_tool_calls_from_response_string_kwargs():
    """Should handle tool_kwargs as JSON string."""
    llm = _make_cortex()

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="search",
                    tool_kwargs='{"query": "snowflake"}',
                ),
            ],
        ),
    )

    selections = llm.get_tool_calls_from_response(response)
    assert selections[0].tool_kwargs == {"query": "snowflake"}


def test_parse_sse_responses_text_only():
    """_parse_sse_responses should handle plain text responses."""
    llm = _make_cortex()

    responses = [
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world!"}}]},
    ]

    result = llm._parse_sse_responses(responses)

    assert result.message.role == MessageRole.ASSISTANT
    assert result.message.content == "Hello world!"
    assert len(result.message.blocks) == 1
    assert isinstance(result.message.blocks[0], TextBlock)


def test_parse_sse_responses_tool_use():
    """_parse_sse_responses should detect tool_use in content_list."""
    llm = _make_cortex()

    responses = [
        {
            "choices": [
                {
                    "delta": {
                        "content": "",
                        "content_list": [
                            {"type": "text", "text": "I'll check that."},
                            {
                                "type": "tool_use",
                                "id": "tu_abc",
                                "name": "get_weather",
                                "input": {"location": "SF"},
                            },
                        ],
                    }
                }
            ]
        }
    ]

    result = llm._parse_sse_responses(responses)

    assert len(result.message.blocks) == 2
    assert isinstance(result.message.blocks[0], TextBlock)
    assert result.message.blocks[0].text == "I'll check that."
    assert isinstance(result.message.blocks[1], ToolCallBlock)
    assert result.message.blocks[1].tool_call_id == "tu_abc"
    assert result.message.blocks[1].tool_name == "get_weather"
    assert result.message.blocks[1].tool_kwargs == {"location": "SF"}


def test_serialize_message_basic():
    """_serialize_message should handle basic text messages."""
    llm = _make_cortex()

    msg = ChatMessage(role=MessageRole.USER, content="Hello")
    result = llm._serialize_message(msg)

    assert result == {"role": "user", "content": "Hello"}


def test_serialize_message_tool_result():
    """_serialize_message should handle tool result messages."""
    llm = _make_cortex()

    msg = ChatMessage(
        role=MessageRole.TOOL,
        content='{"temp": 72}',
        additional_kwargs={"tool_call_id": "call_123"},
    )
    result = llm._serialize_message(msg)

    assert result["role"] == "tool"
    assert result["content"] == '{"temp": 72}'
    assert result["tool_use_id"] == "call_123"


def test_serialize_message_assistant_with_tool_calls():
    """_serialize_message should serialize assistant messages with tool call blocks."""
    llm = _make_cortex()

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            TextBlock(text="Let me check."),
            ToolCallBlock(
                tool_call_id="call_1",
                tool_name="search",
                tool_kwargs={"q": "test"},
            ),
        ],
    )
    result = llm._serialize_message(msg)

    assert result["role"] == "assistant"
    assert isinstance(result["content"], list)
    assert result["content"][0] == {"type": "text", "text": "Let me check."}
    assert result["content"][1]["type"] == "tool_use"
    assert result["content"][1]["id"] == "call_1"
    assert result["content"][1]["name"] == "search"
    assert result["content"][1]["input"] == {"q": "test"}


def test_make_chat_payload_with_tools():
    """_make_chat_payload should include tools and tool_choice in JSON payload."""
    llm = _make_cortex(model="claude-3-5-sonnet")

    tools = [
        {
            "tool_spec": {
                "type": "generic",
                "name": "test",
                "description": "test",
                "input_schema": {},
            }
        }
    ]
    tool_choice = {"type": "auto"}

    payload = llm._make_chat_payload(
        messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        tools=tools,
        tool_choice=tool_choice,
    )

    assert payload["json"]["tools"] == tools
    assert payload["json"]["tool_choice"] == tool_choice


def test_make_chat_payload_without_tools():
    """_make_chat_payload should omit tools/tool_choice when not provided."""
    llm = _make_cortex(model="llama3.2-1b")

    payload = llm._make_chat_payload(
        messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
    )

    assert "tools" not in payload["json"]
    assert "tool_choice" not in payload["json"]
