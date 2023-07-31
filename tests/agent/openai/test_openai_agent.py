from typing import Any, List, Sequence

import pytest
from pytest import MonkeyPatch

from llama_index.agent.openai_agent import OpenAIAgent
from llama_index.chat_engine.types import AgentChatResponse
from llama_index.llms.base import ChatMessage, ChatResponse, MessageRole
from llama_index.llms.mock import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.tools.function_tool import FunctionTool


def mock_chat_completion(*args: Any, **kwargs: Any) -> dict:
    if "functions" in kwargs:
        if not kwargs["functions"]:
            raise ValueError("functions must not be empty")

    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "\n\nThis is a test!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


@pytest.fixture
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


def test_chat_basic(
    add_tool: FunctionTool,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llama_index.llms.openai.completion_with_retry", mock_chat_completion
    )

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"


def test_chat_no_functions(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llama_index.llms.openai.completion_with_retry", mock_chat_completion
    )

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        llm=llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"
