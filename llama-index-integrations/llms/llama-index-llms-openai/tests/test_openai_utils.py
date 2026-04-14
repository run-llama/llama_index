import json
from typing import List

import pytest
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    LogProb,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import (
    ALL_AVAILABLE_MODELS,
    CHAT_MODELS,
    from_openai_completion_logprobs,
    from_openai_message_dicts,
    from_openai_messages,
    from_openai_token_logprob,
    from_openai_token_logprobs,
    is_chat_model,
    is_chatcomp_api_supported,
    is_function_calling_model,
    is_json_schema_supported,
    openai_modelname_to_contextsize,
    to_openai_message_dicts,
    to_openai_tool,
)


@pytest.fixture()
def chat_messages_with_function_calling() -> List[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.USER, content="test question with functions"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{ "location": "Boston, MA"}',
                },
            },
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content='{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            additional_kwargs={
                "tool_call_id": "get_current_weather",
            },
        ),
    ]


@pytest.fixture()
def openai_message_dicts_with_function_calling() -> List[ChatCompletionMessageParam]:
    return [
        {
            "role": "user",
            "content": "test question with functions",
        },
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "get_current_weather",
                "arguments": '{ "location": "Boston, MA"}',
            },
        },
        {
            "role": "tool",
            "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            "tool_call_id": "get_current_weather",
        },
    ]


@pytest.fixture()
def azure_openai_message_dicts_with_function_calling() -> List[ChatCompletionMessage]:
    """
    Taken from:
    - https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling.
    """
    return [
        ChatCompletionMessage(
            role="assistant",
            content=None,
            function_call=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="0123",
                    type="function",
                    function=Function(
                        name="search_hotels",
                        arguments='{\n  "location": "San Diego",\n  "max_price": 300,\n  "features": "beachfront,free breakfast"\n}',
                    ),
                )
            ],
        )
    ]


@pytest.fixture()
def azure_chat_messages_with_function_calling() -> List[ChatMessage]:
    return [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    block_type="tool_call",
                    tool_call_id="0123",
                    tool_name="search_hotels",
                    tool_kwargs='{\n  "location": "San Diego",\n  "max_price": 300,\n  "features": "beachfront,free breakfast"\n}',
                )
            ],
            additional_kwargs={
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        id="0123",
                        type="function",
                        function=Function(
                            name="search_hotels",
                            arguments='{\n  "location": "San Diego",\n  "max_price": 300,\n  "features": "beachfront,free breakfast"\n}',
                        ),
                    )
                ],
            },
        ),
    ]


def test_to_openai_message_dicts_basic_enum() -> None:
    chat_messages = [
        ChatMessage(role=MessageRole.USER, content="test question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="test answer"),
    ]
    openai_messages = to_openai_message_dicts(
        chat_messages,
    )
    assert openai_messages == [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": "test answer"},
    ]


def test_to_openai_message_dicts_basic_string() -> None:
    chat_messages = [
        ChatMessage(role="user", content="test question"),
        ChatMessage(role="assistant", content="test answer"),
    ]
    openai_messages = to_openai_message_dicts(
        chat_messages,
    )
    assert openai_messages == [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": "test answer"},
    ]


def test_to_openai_message_dicts_empty_content() -> None:
    """If neither `tool_calls` nor `function_call` is set, content must not be set to None,
    see: https://platform.openai.com/docs/api-reference/chat/create"""
    chat_messages = [
        ChatMessage(role="user", content="test question"),
        ChatMessage(role="assistant", content=""),
    ]
    openai_messages = to_openai_message_dicts(
        chat_messages,
    )
    assert openai_messages == [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": ""},
    ]


def test_to_openai_message_dicts_function_calling(
    chat_messages_with_function_calling: List[ChatMessage],
    openai_message_dicts_with_function_calling: List[ChatCompletionMessageParam],
) -> None:
    message_dicts = to_openai_message_dicts(
        chat_messages_with_function_calling,
    )
    assert message_dicts == openai_message_dicts_with_function_calling


def test_from_openai_message_dicts_function_calling(
    openai_message_dicts_with_function_calling: List[ChatCompletionMessageParam],
    chat_messages_with_function_calling: List[ChatMessage],
) -> None:
    chat_messages = from_openai_message_dicts(
        openai_message_dicts_with_function_calling
    )  # type: ignore

    # assert attributes match
    for chat_message, chat_message_with_function_calling in zip(
        chat_messages, chat_messages_with_function_calling
    ):
        for key in chat_message.additional_kwargs:
            assert chat_message.additional_kwargs[
                key
            ] == chat_message_with_function_calling.additional_kwargs.get(key, None)
        assert chat_message.content == chat_message_with_function_calling.content
        assert chat_message.role == chat_message_with_function_calling.role


def test_from_openai_messages_function_calling_azure(
    azure_openai_message_dicts_with_function_calling: List[ChatCompletionMessage],
    azure_chat_messages_with_function_calling: List[ChatMessage],
) -> None:
    chat_messages = from_openai_messages(
        azure_openai_message_dicts_with_function_calling,
        ["text"],
    )
    assert chat_messages == azure_chat_messages_with_function_calling


def test_to_openai_tool_with_provided_description() -> None:
    class TestOutput(BaseModel):
        test: str

    tool = to_openai_tool(TestOutput, description="Provided description")
    assert tool == {
        "type": "function",
        "function": {
            "name": "TestOutput",
            "description": "Provided description",
            "parameters": TestOutput.schema(),
        },
    }


def test_to_openai_message_with_pydantic_description() -> None:
    class TestOutput(BaseModel):
        """
        Pydantic description.
        """

        test: str

    tool = to_openai_tool(TestOutput)

    assert tool == {
        "type": "function",
        "function": {
            "name": "TestOutput",
            "description": "Pydantic description.",
            "parameters": TestOutput.schema(),
        },
    }


def test_to_openai_message_dicts_with_content_blocks() -> None:
    chat_message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="test question"),
            ImageBlock(url="https://example.com/image.jpg"),
        ],
    )

    # user messages are converted to blocks
    openai_message = to_openai_message_dicts([chat_message])[0]
    assert openai_message == {
        "role": "user",
        "content": [
            {"type": "text", "text": "test question"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                },
            },
        ],
    }

    chat_message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="test question"),
            ImageBlock(url="https://example.com/image.jpg"),
        ],
    )

    # other messages do not support blocks
    chat_message = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            TextBlock(text="test question"),
            ImageBlock(url="https://example.com/image.jpg"),
        ],
    )

    openai_message = to_openai_message_dicts([chat_message])[0]
    assert openai_message == {
        "role": "assistant",
        "content": "test question",
    }


def test_to_openai_message_dicts_with_content_blocks_with_detail() -> None:
    chat_message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="test question"),
            ImageBlock(url="https://example.com/image.jpg", detail="high"),
        ],
    )

    # user messages are converted to blocks
    openai_message = to_openai_message_dicts([chat_message])[0]
    assert openai_message == {
        "role": "user",
        "content": [
            {"type": "text", "text": "test question"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high",
                },
            },
        ],
    }


def test_from_openai_token_logprob_none_top_logprob() -> None:
    logprob = ChatCompletionTokenLogprob(token="", logprob=1.0, top_logprobs=[])
    logprob.top_logprobs = None
    result: List[LogProb] = from_openai_token_logprob(logprob)
    assert isinstance(result, list)


def test_from_openai_token_logprobs_none_top_logprobs() -> None:
    logprob = ChatCompletionTokenLogprob(token="", logprob=1.0, top_logprobs=[])
    logprob.top_logprobs = None
    result: List[LogProb] = from_openai_token_logprobs([logprob])
    assert isinstance(result, list)


def test_from_openai_completion_logprobs_none_top_logprobs() -> None:
    logprobs = Logprobs(top_logprobs=None)
    result = from_openai_completion_logprobs(logprobs)
    assert isinstance(result, list)


def _build_chat_response(arguments: str) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        id="0123",
                        type="function",
                        function=Function(
                            name="search",
                            arguments=arguments,
                        ),
                    ),
                ],
            },
        ),
    )


def test_get_tool_calls_from_response_returns_empty_arguments_with_invalid_json_arguments() -> (
    None
):
    response = _build_chat_response("INVALID JSON")
    tools = OpenAI().get_tool_calls_from_response(response)
    assert len(tools) == 1
    assert tools[0].tool_kwargs == {}


def test_get_tool_calls_from_response_returns_empty_arguments_with_non_dict_json_input() -> (
    None
):
    response = _build_chat_response("null")
    tools = OpenAI().get_tool_calls_from_response(response)
    assert len(tools) == 1
    assert tools[0].tool_kwargs == {}


def test_get_tool_calls_from_response_returns_arguments_with_dict_json_input() -> None:
    arguments = {"test": 123}
    response = _build_chat_response(json.dumps(arguments))
    tools = OpenAI().get_tool_calls_from_response(response)
    assert len(tools) == 1
    assert tools[0].tool_kwargs == arguments


def test_is_json_schema_supported_supported_models() -> None:
    """Test that supported models return True."""
    supported_models = [
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4.1",
    ]

    for model in supported_models:
        assert is_json_schema_supported(model), f"Model {model} should be supported"


def test_is_json_schema_supported_o1_mini_excluded() -> None:
    """Test that o1-mini models are explicitly excluded."""
    o1_mini_models = [
        "o1-mini",
        "o1-mini-2024-09-12",
    ]

    for model in o1_mini_models:
        assert is_json_schema_supported(model) is False, (
            f"Model {model} should be excluded"
        )


def test_is_json_schema_supported_unsupported_models() -> None:
    """Test that unsupported models return False."""
    unsupported_models = [
        "gpt-3.5-turbo-0613",
        "gpt-4-0613",
        "text-davinci-003",
        "babbage-002",
        "unknown-model",
    ]

    for model in unsupported_models:
        assert is_json_schema_supported(model) is False, (
            f"Model {model} should not be supported"
        )


def test_gpt_5_chat_latest_model_support() -> None:
    """Test that gpt-5-chat-latest is properly supported."""
    model_name = "gpt-5-chat-latest"

    # Test that model is in available models
    assert model_name in ALL_AVAILABLE_MODELS, (
        f"{model_name} should be in ALL_AVAILABLE_MODELS"
    )

    # Test that model is recognized as a chat model
    assert is_chat_model(model_name) is True, (
        f"{model_name} should be recognized as a chat model"
    )

    # Test that model supports function calling
    assert is_function_calling_model(model_name) is True, (
        f"{model_name} should support function calling"
    )

    # Test that model has correct context size
    context_size = openai_modelname_to_contextsize(model_name)
    assert context_size == 128000, (
        f"{model_name} should have 128000 tokens context, got {context_size}"
    )

    # Test that model is in CHAT_MODELS
    assert model_name in CHAT_MODELS, f"{model_name} should be in CHAT_MODELS"


def test_is_chatcomp_api_supported() -> None:
    assert is_chatcomp_api_supported("gpt-5.2")
    assert not is_chatcomp_api_supported("gpt-5.2-pro")
    assert is_chatcomp_api_supported("gpt-5.4")
    assert not is_chatcomp_api_supported("gpt-5.4-pro")


def test_gpt_5_chat_model_support() -> None:
    """Test that gpt-5-chat is properly supported."""
    model_name = "gpt-5-chat"

    assert model_name in ALL_AVAILABLE_MODELS, (
        f"{model_name} should be in ALL_AVAILABLE_MODELS"
    )

    assert is_chat_model(model_name) is True, (
        f"{model_name} should be recognized as a chat model"
    )

    assert is_function_calling_model(model_name) is True, (
        f"{model_name} should support function calling"
    )

    context_size = openai_modelname_to_contextsize(model_name)
    assert context_size == 128000, (
        f"{model_name} should have 128000 tokens context, got {context_size}"
    )

    assert model_name in CHAT_MODELS, f"{model_name} should be in CHAT_MODELS"


def test_gpt_5_4_model_support() -> None:
    """Test that gpt-5.4 is properly supported as a reasoning model."""
    model_name = "gpt-5.4"

    assert model_name in ALL_AVAILABLE_MODELS, (
        f"{model_name} should be in ALL_AVAILABLE_MODELS"
    )

    assert is_chat_model(model_name) is True, (
        f"{model_name} should be recognized as a chat model"
    )

    assert is_function_calling_model(model_name) is True, (
        f"{model_name} should support function calling"
    )

    context_size = openai_modelname_to_contextsize(model_name)
    assert context_size == 1050000, (
        f"{model_name} should have 1050000 tokens context, got {context_size}"
    )

    assert model_name in CHAT_MODELS, f"{model_name} should be in CHAT_MODELS"

    assert is_json_schema_supported(model_name) is True, (
        f"{model_name} should support JSON schema"
    )


def test_gpt_5_4_mini_model_support() -> None:
    """Test that gpt-5.4-mini is properly supported as a reasoning model."""
    model_name = "gpt-5.4-mini"

    assert model_name in ALL_AVAILABLE_MODELS, (
        f"{model_name} should be in ALL_AVAILABLE_MODELS"
    )

    assert is_chat_model(model_name) is True, (
        f"{model_name} should be recognized as a chat model"
    )

    assert is_function_calling_model(model_name) is True, (
        f"{model_name} should support function calling"
    )

    context_size = openai_modelname_to_contextsize(model_name)
    assert context_size == 400000, (
        f"{model_name} should have 400000 tokens context, got {context_size}"
    )

    assert model_name in CHAT_MODELS, f"{model_name} should be in CHAT_MODELS"

    assert is_json_schema_supported(model_name) is True, (
        f"{model_name} should support JSON schema"
    )


def test_gpt_5_4_nano_model_support() -> None:
    """Test that gpt-5.4-nano is properly supported as a reasoning model."""
    model_name = "gpt-5.4-nano"

    assert model_name in ALL_AVAILABLE_MODELS, (
        f"{model_name} should be in ALL_AVAILABLE_MODELS"
    )

    assert is_chat_model(model_name) is True, (
        f"{model_name} should be recognized as a chat model"
    )

    assert is_function_calling_model(model_name) is True, (
        f"{model_name} should support function calling"
    )

    context_size = openai_modelname_to_contextsize(model_name)
    assert context_size == 400000, (
        f"{model_name} should have 400000 tokens context, got {context_size}"
    )

    assert model_name in CHAT_MODELS, f"{model_name} should be in CHAT_MODELS"

    assert is_json_schema_supported(model_name) is True, (
        f"{model_name} should support JSON schema"
    )


def test_gpt_5_4_chat_latest_model_support() -> None:
    """Test that gpt-5.4-chat-latest is properly supported."""
    model_name = "gpt-5.4-chat-latest"

    assert model_name in ALL_AVAILABLE_MODELS, (
        f"{model_name} should be in ALL_AVAILABLE_MODELS"
    )

    assert is_chat_model(model_name) is True, (
        f"{model_name} should be recognized as a chat model"
    )

    assert is_function_calling_model(model_name) is True, (
        f"{model_name} should support function calling"
    )

    context_size = openai_modelname_to_contextsize(model_name)
    assert context_size == 128000, (
        f"{model_name} should have 128000 tokens context, got {context_size}"
    )

    assert model_name in CHAT_MODELS, f"{model_name} should be in CHAT_MODELS"


def test_gpt_5_4_pro_responses_api_only() -> None:
    """Test that gpt-5.4-pro is a Responses API only model."""
    model_name = "gpt-5.4-pro"

    assert not is_chatcomp_api_supported(model_name), (
        f"{model_name} should NOT support Chat Completions API"
    )

    assert model_name not in ALL_AVAILABLE_MODELS, (
        f"{model_name} should NOT be in ALL_AVAILABLE_MODELS (Responses API only)"
    )

    assert is_json_schema_supported(model_name) is True, (
        f"{model_name} should support JSON schema"
    )


def test_responses_api_assistant_text_preserved_with_tool_calls() -> None:
    """Test that assistant text content is included alongside tool calls.

    When an assistant message contains both text blocks and tool calls,
    the text must not be silently dropped.
    Ref: https://github.com/run-llama/llama_index/issues/21124 (bug #1)
    """
    from llama_index.llms.openai.utils import to_openai_responses_message_dict

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            TextBlock(text="I'll search for that information now."),
            ToolCallBlock(
                tool_name="search",
                tool_call_id="call_1",
                tool_kwargs='{"q": "test"}',
            ),
        ],
    )

    result = to_openai_responses_message_dict(msg, model="o3-mini")
    assert isinstance(result, list)

    text_items = [
        item
        for item in result
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    tool_items = [
        item
        for item in result
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]

    assert len(text_items) == 1, "Assistant text content should be preserved"
    assert text_items[0]["content"] == "I'll search for that information now."
    assert len(tool_items) == 1, "Tool call should be preserved"
    assert tool_items[0]["name"] == "search"


def test_responses_api_tool_only_no_empty_text() -> None:
    """Test that tool-call-only messages don't include an empty text item."""
    from llama_index.llms.openai.utils import to_openai_responses_message_dict

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_name="search",
                tool_call_id="call_1",
                tool_kwargs='{"q": "test"}',
            ),
        ],
    )

    result = to_openai_responses_message_dict(msg, model="o3-mini")
    assert isinstance(result, list)

    text_items = [
        item
        for item in result
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    assert len(text_items) == 0, "No text item should be emitted for tool-only messages"


def test_responses_api_tool_kwargs_serialized_to_json_string() -> None:
    """Test that dict tool_kwargs are serialized to JSON strings.

    The OpenAI Responses API expects 'arguments' to be a JSON string,
    but ToolCallBlock.tool_kwargs can be a dict.
    Ref: https://github.com/run-llama/llama_index/issues/21124 (bug #6)
    """
    from llama_index.llms.openai.utils import to_openai_responses_message_dict

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_name="get_weather",
                tool_call_id="call_2",
                tool_kwargs={"location": "Boston", "unit": "celsius"},
            ),
        ],
    )

    result = to_openai_responses_message_dict(msg, model="gpt-5.4")
    assert isinstance(result, list)

    tool_item = [
        item
        for item in result
        if isinstance(item, dict) and item.get("type") == "function_call"
    ][0]
    assert isinstance(tool_item["arguments"], str), "arguments must be a JSON string"
    assert json.loads(tool_item["arguments"]) == {
        "location": "Boston",
        "unit": "celsius",
    }


def test_responses_api_tool_kwargs_string_passthrough() -> None:
    """Test that string tool_kwargs are passed through unchanged."""
    from llama_index.llms.openai.utils import to_openai_responses_message_dict

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_name="search",
                tool_call_id="call_3",
                tool_kwargs='{"q": "test"}',
            ),
        ],
    )

    result = to_openai_responses_message_dict(msg, model="gpt-5.4")
    assert isinstance(result, list)

    tool_item = [
        item
        for item in result
        if isinstance(item, dict) and item.get("type") == "function_call"
    ][0]
    assert tool_item["arguments"] == '{"q": "test"}'


def test_chat_completions_tool_kwargs_serialized_to_json_string() -> None:
    """Test that dict tool_kwargs are serialized to JSON strings in Chat Completions API.

    The OpenAI Chat Completions API expects 'arguments' to be a JSON string,
    but ToolCallBlock.tool_kwargs can be a dict. This caused 400 BadRequestError
    when using mixed LLM providers (e.g., Anthropic orchestrator -> OpenAI sub-agent).
    Ref: https://github.com/run-llama/llama_index/issues/21378
    """
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_name="get_weather",
                tool_call_id="call_123",
                tool_kwargs={"location": "Boston", "unit": "celsius"},
            ),
        ],
    )

    result = to_openai_message_dict(msg)
    # result should have tool_calls with function.arguments as JSON string
    tool_calls = result.get("tool_calls", [])
    assert len(tool_calls) == 1
    function = tool_calls[0]["function"]
    assert isinstance(function["arguments"], str), "arguments must be a JSON string"
    assert json.loads(function["arguments"]) == {
        "location": "Boston",
        "unit": "celsius",
    }


def test_chat_completions_tool_kwargs_string_passthrough() -> None:
    """Test that string tool_kwargs are passed through unchanged in Chat Completions API."""
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_name="search",
                tool_call_id="call_456",
                tool_kwargs='{"q": "test"}',
            ),
        ],
    )

    result = to_openai_message_dict(msg)
    tool_calls = result.get("tool_calls", [])
    assert len(tool_calls) == 1
    function = tool_calls[0]["function"]
    assert function["arguments"] == '{"q": "test"}'
