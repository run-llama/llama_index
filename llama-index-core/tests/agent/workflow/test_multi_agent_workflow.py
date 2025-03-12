from typing import Any, List

import pytest
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.agent.workflow.multi_agent_workflow import AgentWorkflow
from llama_index.core.agent.workflow.react_agent import ReActAgent
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    LLMMetadata,
    MessageRole,
    MockLLM,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, ToolSelection
from llama_index.core.workflow import WorkflowRuntimeError


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
def empty_calculator_agent():
    return ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant.",
        tools=[
            FunctionTool.from_defaults(fn=add),
            FunctionTool.from_defaults(fn=subtract),
        ],
        llm=MockLLM(responses=[]),
    )


@pytest.fixture()
def retriever_agent():
    return FunctionAgent(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Let me help you with that calculation. I'll hand this off to the calculator.",
                    additional_kwargs={
                        "tool_calls": [
                            ToolSelection(
                                tool_id="one",
                                tool_name="handoff",
                                tool_kwargs={
                                    "to_agent": "calculator",
                                    "reason": "This requires arithmetic operations.",
                                },
                            )
                        ]
                    },
                ),
            ],
        ),
    )


@pytest.fixture()
def empty_retriever_agent():
    return FunctionAgent(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        llm=MockLLM(
            responses=[],
        ),
    )


@pytest.mark.asyncio()
async def test_basic_workflow(calculator_agent, retriever_agent):
    """Test basic workflow initialization and validation."""
    workflow = AgentWorkflow(
        agents=[calculator_agent, retriever_agent],
        root_agent="retriever",
    )

    assert workflow.root_agent == retriever_agent.name
    assert len(workflow.agents) == 2
    assert "calculator" in workflow.agents
    assert "retriever" in workflow.agents


@pytest.mark.asyncio()
async def test_workflow_requires_root_agent():
    """Test that workflow requires exactly one root agent."""
    with pytest.raises(ValueError, match="Exactly one root agent must be provided"):
        AgentWorkflow(
            agents=[
                FunctionAgent(
                    name="agent1",
                    description="test",
                    llm=MockLLM(
                        responses=[
                            ChatMessage(role=MessageRole.ASSISTANT, content="test"),
                        ]
                    ),
                ),
                ReActAgent(
                    name="agent2",
                    description="test",
                    llm=MockLLM(
                        responses=[
                            ChatMessage(role=MessageRole.ASSISTANT, content="test"),
                        ]
                    ),
                ),
            ]
        )


@pytest.mark.asyncio()
async def test_workflow_execution(calculator_agent, retriever_agent):
    """Test basic workflow execution with agent handoff."""
    workflow = AgentWorkflow(
        agents=[calculator_agent, retriever_agent],
        root_agent="retriever",
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = workflow.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler

    # Verify we got events indicating handoff and calculation
    assert any(
        ev.current_agent_name == "retriever"
        if hasattr(ev, "current_agent_name")
        else False
        for ev in events
    )
    assert any(
        ev.current_agent_name == "calculator"
        if hasattr(ev, "current_agent_name")
        else False
        for ev in events
    )
    assert "8" in str(response.response)


@pytest.mark.asyncio()
async def test_workflow_execution_empty(empty_calculator_agent, retriever_agent):
    """Test basic workflow execution with agent handoff."""
    workflow = AgentWorkflow(
        agents=[empty_calculator_agent, retriever_agent],
        root_agent="retriever",
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = workflow.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    with pytest.raises(WorkflowRuntimeError, match="Got empty message"):
        await handler


@pytest.mark.asyncio()
async def test_workflow_handoff_empty(calculator_agent, empty_retriever_agent):
    """Test basic workflow execution with agent handoff."""
    workflow = AgentWorkflow(
        agents=[calculator_agent, empty_retriever_agent],
        root_agent="retriever",
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = workflow.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler
    assert response.response.content is None


@pytest.mark.asyncio()
async def test_invalid_handoff():
    """Test handling of invalid agent handoff."""
    agent1 = FunctionAgent(
        name="agent1",
        description="test",
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="handoff invalid_agent Because reasons",
                    additional_kwargs={
                        "tool_calls": [
                            ToolSelection(
                                tool_id="one",
                                tool_name="handoff",
                                tool_kwargs={
                                    "to_agent": "invalid_agent",
                                    "reason": "Because reasons",
                                },
                            )
                        ]
                    },
                ),
                ChatMessage(role=MessageRole.ASSISTANT, content="guess im stuck here"),
            ],
        ),
    )

    agent2 = FunctionAgent(
        **agent1.model_dump(exclude={"llm"}), llm=MockLLM(responses=[])
    )
    agent2.name = "agent2"

    workflow = AgentWorkflow(
        agents=[agent1, agent2],
        root_agent="agent1",
    )

    handler = workflow.run(user_msg="test")
    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler
    assert "Agent invalid_agent not found" in str(events)


@pytest.mark.asyncio()
async def test_workflow_with_state():
    """Test workflow with state management."""
    agent = FunctionAgent(
        name="agent",
        description="test",
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT, content="Current state processed"
                )
            ],
        ),
    )

    workflow = AgentWorkflow(
        agents=[agent],
        initial_state={"counter": 0},
        state_prompt="Current state: {state}. User message: {msg}",
    )

    handler = workflow.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None
