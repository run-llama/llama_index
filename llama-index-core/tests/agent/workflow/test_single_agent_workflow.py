from typing import List

import pytest

from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentInput
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.errors import WorkflowRuntimeError


def _response_generator_from_list(responses: List[ChatMessage]):
    """Helper to create a response generator from a list of responses."""
    index = 0

    def generator(messages: List[ChatMessage]) -> ChatMessage:
        nonlocal index
        if not responses:
            return ChatMessage(role=MessageRole.ASSISTANT, content=None)
        msg = responses[index]
        index = (index + 1) % len(responses)
        return msg

    return generator


@pytest.fixture()
def function_agent():
    return FunctionAgent(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Success with the FunctionAgent",
                    )
                ]
            )
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
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content='Thought: I need to add these numbers\nAction: add\nAction Input: {"a": 5, "b": 3}\n',
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=r"Thought: The result is 8\Answer: The sum is 8",
                    ),
                ]
            )
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
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
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
            )
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
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
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
            )
        ),
    )

    # Default max iterations is 20
    with pytest.raises(WorkflowRuntimeError, match="Either something went wrong"):
        _ = await agent.run(user_msg="test")

    # Set max iterations to 101 to avoid error
    _ = agent.run(user_msg="test", max_iterations=101)


@pytest.mark.asyncio
async def test_function_agent_with_mock_function_calling_llm():
    """Test that FunctionAgent works with MockFunctionCallingLLM."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # Create a MockFunctionCallingLLM that will return a simple response
    llm = MockFunctionCallingLLM(max_tokens=200)

    agent = FunctionAgent(
        name="test_agent",
        description="A test agent",
        system_prompt="You are a helpful assistant.",
        tools=[FunctionTool.from_defaults(fn=multiply)],
        llm=llm,
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = agent.run(user_msg="Hello, can you help me?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler

    # Verify the agent ran successfully with MockFunctionCallingLLM
    assert response.response is not None
    # Verify that we got some events during execution
    assert len(events) > 0


@pytest.mark.asyncio
async def test_function_agent_with_context_and_chat_message():
    """Test FunctionAgent with explicit Context and ChatMessage input following the full adoption pattern."""
    from llama_index.core.llms import TextBlock
    from llama_index.core.workflow import Context

    AGENT_SYSTEM_PROMPT = (
        "You are a mathematical assistant that helps with calculations."
    )

    def noop() -> str:
        """A no-op function for testing."""
        return "noop executed"

    # Step 1: Construct FunctionAgent
    llm = MockFunctionCallingLLM()
    function_tools = [FunctionTool.from_defaults(fn=noop)]
    constructor_kwargs = {
        "llm": llm,
        "tools": function_tools,
        "system_prompt": AGENT_SYSTEM_PROMPT,
    }
    # Perhaps additional kwargs mutations would go here.
    agent = FunctionAgent(**constructor_kwargs)

    # Step 2: Construct context
    ctx = Context(agent)

    # Step 3: Construct user message as ChatMessage
    user_message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Can you help me with a calculation?"),
            TextBlock(text="It's one plus one."),
        ],
    )

    # Step 4: Trigger agent run with ctx
    handler = agent.run(
        user_msg=user_message,
        ctx=ctx,
    )

    # Step 5: Await response
    response = await handler

    # Verify the response contains meaningful content
    assert response is not None
    # The response should not just be the system prompt
    assert response != AGENT_SYSTEM_PROMPT
    # Verify we got a proper response from the LLM
    assert len(str(response)) > 0
    # Verify that both TextBlock contents from the user message are in the response
    response_str = str(response)
    assert "Can you help me with a calculation?" in response_str
    assert "It's one plus one." in response_str
