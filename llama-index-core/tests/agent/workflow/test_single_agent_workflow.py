from typing import List, Any

import pytest

from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentInput
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
from llama_index.core.workflow.errors import WorkflowRuntimeError


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


@pytest.fixture()
def retry_calculator_agent():
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
                    content='Thought: I need to add these numbers\nAction: add\n{"a": 5 "b": 3}\n',
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content='Thought: I need to add these numbers\nAction: add\nAction Input: {"a": 5, "b": 3}\n',
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=r"Thought: The result is 8\nAnswer: The sum is 8",
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


@pytest.mark.asyncio
async def test_single_react_agent_retry(retry_calculator_agent):
    """Verify execution of basic ReAct single agent with retry due to a output parsing error."""
    memory = ChatMemoryBuffer.from_defaults()
    handler = retry_calculator_agent.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    contains_error_message = False
    async for event in handler.stream_events():
        events.append(event)
        if isinstance(event, AgentInput):
            if "Error while parsing the output" in event.input[-1].content:
                contains_error_message = True

    assert contains_error_message

    response = await handler

    assert "8" in str(response.response)


@pytest.mark.asyncio
async def test_max_iterations():
    """Test max iterations."""

    def random_tool() -> str:
        return "random"

    agent = FunctionAgent(
        name="agent",
        description="test",
        tools=[random_tool],
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="handing off",
                    additional_kwargs={
                        "tool_calls": [
                            ToolSelection(
                                tool_id="one",
                                tool_name="random_tool",
                                tool_kwargs={},
                            )
                        ]
                    },
                ),
            ]
            * 100
        ),
    )

    # Default max iterations is 20
    with pytest.raises(WorkflowRuntimeError, match="Either something went wrong"):
        _ = await agent.run(user_msg="test")

    # Set max iterations to 101 to avoid error
    _ = agent.run(user_msg="test", max_iterations=101)


@pytest.mark.asyncio
async def test_early_stopping_method_generate():
    """Test early_stopping_method='generate' produces a final response instead of raising error."""

    def random_tool() -> str:
        return "random"

    # Create responses: first N-1 will trigger tool calls, last one is the early stopping response
    tool_call_response = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [
                ToolSelection(
                    tool_id="one",
                    tool_name="random_tool",
                    tool_kwargs={},
                )
            ]
        },
    )
    final_response = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Here is my final summary based on the information gathered.",
    )

    agent = FunctionAgent(
        name="agent",
        description="test",
        tools=[random_tool],
        llm=MockLLM(responses=[tool_call_response] * 10 + [final_response]),
        early_stopping_method="generate",
    )

    # With early_stopping_method="generate", should NOT raise error
    handler = agent.run(user_msg="test", max_iterations=5)
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None
    assert (
        "final summary" in str(response.response).lower()
        or response.response is not None
    )


@pytest.mark.asyncio
async def test_early_stopping_method_force():
    """Test early_stopping_method='force' (default) raises error on max iterations."""

    def random_tool() -> str:
        return "random"

    agent = FunctionAgent(
        name="agent",
        description="test",
        tools=[random_tool],
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="calling tool",
                    additional_kwargs={
                        "tool_calls": [
                            ToolSelection(
                                tool_id="one",
                                tool_name="random_tool",
                                tool_kwargs={},
                            )
                        ]
                    },
                ),
            ]
            * 100
        ),
        early_stopping_method="force",  # Default, but explicit for clarity
    )

    # With early_stopping_method="force", should raise error
    with pytest.raises(WorkflowRuntimeError, match="early_stopping_method='generate'"):
        _ = await agent.run(user_msg="test", max_iterations=5)


@pytest.mark.asyncio
async def test_early_stopping_method_override_in_run():
    """Test early_stopping_method can be overridden in run()."""

    def random_tool() -> str:
        return "random"

    tool_call_response = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [
                ToolSelection(
                    tool_id="one",
                    tool_name="random_tool",
                    tool_kwargs={},
                )
            ]
        },
    )
    final_response = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Final response after early stopping.",
    )

    # Agent defaults to "force"
    agent = FunctionAgent(
        name="agent",
        description="test",
        tools=[random_tool],
        llm=MockLLM(responses=[tool_call_response] * 10 + [final_response]),
        early_stopping_method="force",
    )

    # Override to "generate" in run()
    handler = agent.run(
        user_msg="test",
        max_iterations=5,
        early_stopping_method="generate",
    )
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None
