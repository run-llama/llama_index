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
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import (
    from_openai_completion_logprobs,
    from_openai_message_dicts,
    from_openai_messages,
    from_openai_token_logprob,
    from_openai_token_logprobs,
    is_json_schema_supported,
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
            content=None,
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
                    "detail": "auto",
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
        assert is_json_schema_supported(model) is True, (
            f"Model {model} should be supported"
        )


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
