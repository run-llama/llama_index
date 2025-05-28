import os
from typing import Sequence, Optional, List
from unittest import mock

import pytest
from cohere import ChatbotMessage, UserMessage, NonStreamedChatResponse, ToolCall

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from llama_index.core.llms.mock import MockLLM
from llama_index.core.tools import FunctionTool


from llama_index.llms.cohere import Cohere, DocumentMessage, is_cohere_model


def test_is_cohere():
    assert is_cohere_model(Cohere(api_key="mario"))
    assert not is_cohere_model(MockLLM())


@pytest.mark.skipif(
    os.getenv("COHERE_API_KEY") is None, reason="COHERE_API_KEY is not set"
)
def test_tool_required():
    llm = Cohere(
        api_key=os.getenv("COHERE_API_KEY"),
        model="command-r7b-12-2024",
        temperature=0.3,
    )
    result = llm.chat_with_tools(
        tools=[search_tool],
        user_msg="What is the capital of France? Respond simply",
        tool_required=True,
    )
    assert "tool_calls" in result.message.additional_kwargs
    assert len(result.message.additional_kwargs["tool_calls"]) == 1
    assert result.message.additional_kwargs["tool_calls"][0].name == "search_tool"


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in Cohere.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    "messages,expected_chat_history,expected_documents,expected_message",  # noqa: PT006
    [
        pytest.param(
            [ChatMessage(content="Hello", role=MessageRole.USER)],
            [],
            None,
            "Hello",
            id="single user message",
        ),
        pytest.param(
            [
                ChatMessage(content="Earliest message", role=MessageRole.USER),
                ChatMessage(content="Latest message", role=MessageRole.USER),
            ],
            [{"message": "Earliest message", "role": "User"}],
            None,
            "Latest message",
            id="messages with chat history",
        ),
        pytest.param(
            [
                ChatMessage(content="Earliest message", role=MessageRole.USER),
                DocumentMessage(content="Document content"),
                ChatMessage(content="Latest message", role=MessageRole.USER),
            ],
            [{"message": "Earliest message", "role": "User"}],
            [{"text": "Document content"}],
            "Latest message",
            id="messages with chat history",
        ),
    ],
)
def test_chat(
    messages: Sequence[ChatMessage],
    expected_chat_history: Optional[List],
    expected_documents: Optional[List],
    expected_message: str,
):
    # Mock the API client.
    with mock.patch("llama_index.llms.cohere.base.cohere.Client", autospec=True):
        llm = Cohere(api_key="dummy", temperature=0.3)
    # Mock the API response.
    llm._client.chat.return_value = NonStreamedChatResponse(text="Placeholder reply")
    expected = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Placeholder reply"),
        raw=llm._client.chat.return_value.__dict__,
    )

    actual = llm.chat(messages)
    assert expected.raw == actual.raw
    assert expected.message.content == actual.message.content
    assert expected.additional_kwargs == actual.additional_kwargs
    # Assert that the mocked API client was called in the expected way.

    if expected_documents:
        llm._client.chat.assert_called_once_with(
            chat_history=expected_chat_history,
            documents=expected_documents,
            message=expected_message,
            model="command-r",
            temperature=0.3,
        )
    else:
        llm._client.chat.assert_called_once_with(
            chat_history=expected_chat_history,
            message=expected_message,
            model="command-r",
            temperature=0.3,
        )


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required is correctly passed to the API request when True."""
    with mock.patch("llama_index.llms.cohere.base.cohere.Client", autospec=True):
        llm = Cohere(api_key="dummy", temperature=0.3)

    # Test with tool_required=True
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

    assert "force_single_step" in result
    assert result["force_single_step"]
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    with mock.patch("llama_index.llms.cohere.base.cohere.Client", autospec=True):
        llm = Cohere(api_key="dummy", temperature=0.3)

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
    )

    assert "force_single_step" not in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_invoke_tool_calls() -> None:
    with mock.patch("llama_index.llms.cohere.base.cohere.Client", autospec=True):
        llm = Cohere(api_key="dummy", temperature=0.3)

    def multiply(a: int, b: int) -> int:
        """Multiple two integers and returns the result integer."""
        return a * b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    add_tool = FunctionTool.from_defaults(fn=add)

    llm._client.chat.return_value = {
        "text": "I will use the multiply tool to calculate 3 times 4, then use the add tool to add 5 to the answer.",
        "generation_id": "26077c34-49e7-4c0b-941e-602ed684aa64",
        "finish_reason": "COMPLETE",
        "tool_calls": [ToolCall(name="multiply", parameters={"a": 3, "b": 4})],
        "chat_history": [
            UserMessage(
                message="What is 3 times 4 plus 5?",
            ),
            ChatbotMessage(
                message="I will use the multiply tool to calculate 3 times 4, then use the add tool to add 5 to the answer.",
                tool_calls=[ToolCall(name="multiply", parameters={"a": 3, "b": 4})],
            ),
        ],
        "prompt": None,
        "response_id": "some-id",
    }

    result = llm.chat_with_tools(
        tools=[multiply_tool, add_tool],
        user_msg="What is 3 times 4 plus 5?",
        allow_parallel_tool_calls=True,
    )
    assert isinstance(result, ChatResponse)
    additional_kwargs = result.message.additional_kwargs
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0].name == "multiply"
    assert additional_kwargs["tool_calls"][0].parameters == {
        "a": 3,
        "b": 4,
    }
