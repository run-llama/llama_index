from typing import Any, List, Sequence

import pytest

from llama_index.agent.react.base import ReActAgent
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.llms.mock import MockLLM
from llama_index.tools.function_tool import FunctionTool


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

        super().__init__()

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
) -> None:
    mock_llm = MockChatLLM(
        responses=[
            ChatMessage(
                content=MOCK_ACTION_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
            ChatMessage(
                content=MOCK_FINAL_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
        ]
    )

    agent = ReActAgent.from_tools(
        tools=[add_tool],
        llm=mock_llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "2"

    chat_history = agent.chat_history
    assert chat_history == [
        ChatMessage(
            content="What is 1 + 1?",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content="2",
            role=MessageRole.ASSISTANT,
        ),
    ]


@pytest.mark.asyncio
async def test_achat_basic(
    add_tool: FunctionTool,
) -> None:
    mock_llm = MockChatLLM(
        responses=[
            ChatMessage(
                content=MOCK_ACTION_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
            ChatMessage(
                content=MOCK_FINAL_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
        ]
    )

    agent = ReActAgent.from_tools(
        tools=[add_tool],
        llm=mock_llm,
    )
    response = await agent.achat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "2"

    chat_history = agent.chat_history
    assert chat_history == [
        ChatMessage(
            content="What is 1 + 1?",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content="2",
            role=MessageRole.ASSISTANT,
        ),
    ]


class MockStreamChatLLM(MockLLM):
    def __init__(self, responses: List[ChatMessage]) -> None:
        self._i = 0  # call counter, determines which response to return
        self._responses = responses  # list of responses to return

        super().__init__()

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        del messages  # unused
        full_message = self._responses[self._i]
        self._i += 1

        role = full_message.role
        full_text = full_message.content or ""

        text_so_far = ""
        for token in full_text:
            text_so_far += token
            message = ChatMessage(
                content=text_so_far,
                role=role,
            )
            yield ChatResponse(
                message=message,
                delta=token,
            )


MOCK_STREAM_FINAL_RESPONSE = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: 2 is the final answer.
"""


def test_stream_chat_basic(
    add_tool: FunctionTool,
) -> None:
    mock_llm = MockStreamChatLLM(
        responses=[
            ChatMessage(
                content=MOCK_ACTION_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
            ChatMessage(
                content=MOCK_STREAM_FINAL_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
        ]
    )

    agent = ReActAgent.from_tools(
        tools=[add_tool],
        llm=mock_llm,
    )
    response = agent.stream_chat("What is 1 + 1?")
    assert isinstance(response, StreamingAgentChatResponse)

    text_so_far = ""
    counter = 0
    for delta in response.response_gen:
        print(f"{delta=}")
        text_so_far += delta
        counter += 1
    print(f"{MOCK_STREAM_FINAL_RESPONSE=}")
    expected_answer = MOCK_STREAM_FINAL_RESPONSE.split("Answer: ")[-1]
    print(f"{expected_answer=}")
    print(f"{text_so_far}")
    assert text_so_far == expected_answer
    assert counter == len(expected_answer)

    assert agent.chat_history == [
        ChatMessage(
            content="What is 1 + 1?",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content="2 is the final answer.\n",
            role=MessageRole.ASSISTANT,
        ),
    ]


@pytest.mark.asyncio
async def test_astream_chat_basic(
    add_tool: FunctionTool,
) -> None:
    mock_llm = MockStreamChatLLM(
        responses=[
            ChatMessage(
                content=MOCK_ACTION_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
            ChatMessage(
                content=MOCK_STREAM_FINAL_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
        ]
    )

    agent = ReActAgent.from_tools(
        tools=[add_tool],
        llm=mock_llm,
    )
    response = await agent.astream_chat("What is 1 + 1?")
    assert isinstance(response, StreamingAgentChatResponse)

    text_so_far = ""
    counter = 0
    for delta in response.response_gen:
        text_so_far += delta
        counter += 1

    expected_answer = MOCK_STREAM_FINAL_RESPONSE.split("Answer: ")[-1]
    assert text_so_far == expected_answer
    assert counter == len(expected_answer)

    assert agent.chat_history == [
        ChatMessage(
            content="What is 1 + 1?",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content="2 is the final answer.\n",
            role=MessageRole.ASSISTANT,
        ),
    ]
