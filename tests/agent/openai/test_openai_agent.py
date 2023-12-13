from typing import Any, List, Sequence
from unittest.mock import MagicMock, patch

import pytest
from llama_index.agent.openai_agent import OpenAIAgent, call_tool_with_error_handling
from llama_index.chat_engine.types import AgentChatResponse
from llama_index.llms.mock import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.types import ChatMessage, ChatResponse
from llama_index.tools.function_tool import FunctionTool
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage


def mock_chat_completion(*args: Any, **kwargs: Any) -> ChatCompletion:
    if "functions" in kwargs:
        if not kwargs["functions"]:
            raise ValueError("functions must not be empty")

    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


@pytest.fixture()
def add_tool() -> FunctionTool:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    return FunctionTool.from_defaults(fn=add)


class MockChatLLM(MockLLM):
    def __init__(self, responses: List[ChatMessage]) -> None:
        self._i = 0  # call counter, determines which response to return
        self._responses = responses  # list of responses to return

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        del messages  # unused
        response = ChatResponse(
            message=self._responses[self._i],
        )
        self._i += 1
        return response


MOCK_ACTION_RESPONSE = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: {"a": 1, "b": 1}
"""

MOCK_FINAL_RESPONSE = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: 2
"""


@patch("llama_index.llms.openai.SyncOpenAI")
def test_chat_basic(MockSyncOpenAI: MagicMock, add_tool: FunctionTool) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"


@patch("llama_index.llms.openai.SyncOpenAI")
def test_chat_no_functions(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        llm=llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"


def test_call_tool_with_error_handling() -> None:
    """Test call tool with error handling."""

    def _add(a: int, b: int) -> int:
        return a + b

    tool = FunctionTool.from_defaults(fn=_add)

    output = call_tool_with_error_handling(
        tool, {"a": 1, "b": 1}, error_message="Error!"
    )
    assert output.content == "2"

    # try error
    output = call_tool_with_error_handling(
        tool, {"a": "1", "b": 1}, error_message="Error!"
    )
    assert output.content == "Error!"
