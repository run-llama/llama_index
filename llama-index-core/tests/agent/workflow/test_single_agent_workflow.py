from typing import List, Any

import pytest

from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    LLMMetadata,
    ChatResponseAsyncGen,
    ChatResponse,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool


class MockLLM(MockLLM):
    def __init__(self, responses: List[ChatMessage]):
        super().__init__()
        self._responses = responses
        self._response_index = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    async def astream_chat_with_tools(
        self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    def get_tool_calls_from_response(
        self, response: ChatResponse, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])


@pytest.fixture()
def function_agent():
    return FunctionAgent(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT, content="Success with the FunctionAgent"
                )
            ],
        ),
    )


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


@pytest.fixture()
def calculator_agent():
    return ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant.",
        tools=[
            FunctionTool.from_defaults(fn=add),
            FunctionTool.from_defaults(fn=subtract),
        ],
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content='Thought: I need to add these numbers\nAction: add\nAction Input: {"a": 5, "b": 3}\n',
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=r"Thought: The result is 8\Answer: The sum is 8",
                ),
            ]
        ),
    )


@pytest.mark.asyncio
async def test_single_function_agent(function_agent):
    """Test single agent with state management."""
    handler = function_agent.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)


@pytest.mark.asyncio
async def test_single_react_agent(calculator_agent):
    """Verify execution of basic ReAct single agent."""
    memory = ChatMemoryBuffer.from_defaults()
    handler = calculator_agent.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler

    assert "8" in str(response.response)
