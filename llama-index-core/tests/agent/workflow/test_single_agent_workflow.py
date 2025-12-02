from typing import List, Any

import pytest

from llama_index.core.agent.workflow import (
    FunctionAgent,
    ReActAgent,
    AgentInput,
    AgentOutput,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    DocumentBlock,
    TextBlock,
    ThinkingBlock,
    LLMMetadata,
    ChatResponseAsyncGen,
    ChatResponse,
    MessageRole,
)
from llama_index.core.llms import MockLLM, MockEchoLLM
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
async def test_mock_echo_llm():
    """Test the echo llm."""
    max_block_size = 50
    agent = FunctionAgent(
        name="agent",
        description="testing echo",
        tools=[
            FunctionTool.from_defaults(fn=add),
        ],
        llm=MockEchoLLM(max_block_size=max_block_size),
    )

    # Pass in some simple chat messages to verify we get the same thing back.
    result = await agent.run(
        user_msg=ChatMessage(
            role=MessageRole.USER,
            content="Hello, AI!",
        )
    )

    assert "Hello, AI!" in str(result)

    # Pass in multiple Text blocks:
    result = await agent.run(
        user_msg=ChatMessage(
            role=MessageRole.USER,
            content=[
                TextBlock(text="This is block one."),
                TextBlock(text="This is block two."),
            ],
        )
    )
    assert isinstance(result, AgentOutput)
    assert isinstance(result.response, ChatMessage)
    assert len(result.response.blocks) == 1
    assert "<TextBlock>This is block one.</TextBlock>" in result.response.blocks[0].text
    assert "<TextBlock>This is block two.</TextBlock>" in result.response.blocks[0].text

    # Pass in a String and a DocumentBlock.
    result = await agent.run(
        user_msg=ChatMessage(
            role=MessageRole.USER,
            content=[
                TextBlock(text="This is a text block."),
                DocumentBlock(
                    title="Doc1", data=b"Sample data", document_mimetype="text/plain"
                ),
            ],
        )
    )
    # result is AgentOutput with a response of ChatMessage with multiple blocks
    assert isinstance(result, AgentOutput)
    assert isinstance(result.response, ChatMessage)
    assert len(result.response.blocks) == 1
    assert (
        "<TextBlock>This is a text block.</TextBlock>" in result.response.blocks[0].text
    )
    assert (
        "<DocumentBlock title='Doc1' mimetype='text/plain'>Sample data</DocumentBlock>"
        in result.response.blocks[0].text
    )

    # Pass in a string, a ThinkingBlock, and then another string.
    result = await agent.run(
        user_msg=ChatMessage(
            role=MessageRole.USER,
            content=[
                TextBlock(text="Starting with a text block."),
                ThinkingBlock(),
                TextBlock(text="Ending with another text block."),
            ],
        )
    )
    assert isinstance(result, AgentOutput)
    assert isinstance(result.response, ChatMessage)
    assert len(result.response.blocks) == 1
    assert (
        "<TextBlock>Starting with a text block.</TextBlock>"
        in result.response.blocks[0].text
    )
    assert (
        "<UnsupportedBlock type=ThinkingBlock></UnsupportedBlock>"
        in result.response.blocks[0].text
    )
    assert (
        "<TextBlock>Ending with another text block.</TextBlock>"
        in result.response.blocks[0].text
    )

    # Pass in an empty content list.
    result = await agent.run(
        user_msg=ChatMessage(
            role=MessageRole.USER,
            content=[],
        )
    )
    assert isinstance(result, AgentOutput)
    assert isinstance(result.response, ChatMessage)
    assert len(result.response.blocks) == 1
    assert "</empty>" in result.response.blocks[0].text

    # Pass in some history messages without a user_msg. Only the latest is used:
    result = await agent.run(
        chat_history=[
            ChatMessage(
                role=MessageRole.USER,
                content="First message.",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="Second message.",
            ),
        ]
    )
    assert isinstance(result, AgentOutput)
    assert isinstance(result.response, ChatMessage)
    assert "First message." not in result.response.blocks[0].text
    assert "<TextBlock>Second message.</TextBlock>" in result.response.blocks[0].text

    # Add a test which clips content at the max block size:
    long_text = "A" * (max_block_size * 2)
    result = await agent.run(
        user_msg=ChatMessage(
            role=MessageRole.USER,
            content=long_text,
        )
    )
    assert isinstance(result, AgentOutput)
    assert isinstance(result.response, ChatMessage)
    assert len(result.response.blocks) == 1
    assert long_text not in result.response.blocks[0].text
    assert (
        f"<TextBlock>{'A' * max_block_size}</TextBlock>"
        in result.response.blocks[0].text
    )
