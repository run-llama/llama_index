from typing import Any, List
import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.agent.multi_agent.multi_agent_workflow import MultiAgentWorkflow
from llama_index.core.agent.multi_agent.agent_config import AgentConfig, AgentMode
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ChatResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.tools import FunctionTool, ToolSelection
from llama_index.core.memory import ChatMemoryBuffer


class MockLLM(MockLLM):
    def __init__(self, responses: List[ChatMessage], is_function_calling: bool = False):
        super().__init__()
        self._responses = responses
        self._response_index = 0
        self._is_function_calling = is_function_calling

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=self._is_function_calling)

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = self._responses[self._response_index]
        self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            yield ChatResponse(
                message=response_msg,
                delta=response_msg.content,
                raw={"content": response_msg.content},
            )

        return _gen()

    async def astream_chat_with_tools(
        self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = self._responses[self._response_index]
        self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
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
    return AgentConfig(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant.",
        mode=AgentMode.REACT,
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
def retriever_agent():
    return AgentConfig(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        is_entrypoint_agent=True,
        mode=AgentMode.FUNCTION,
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
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="handoff calculator Because this requires arithmetic operations.",
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
            is_function_calling=True,
        ),
    )


@pytest.mark.asyncio()
async def test_basic_workflow(calculator_agent, retriever_agent):
    """Test basic workflow initialization and validation."""
    workflow = MultiAgentWorkflow(
        agent_configs=[calculator_agent, retriever_agent],
    )

    assert workflow.root_agent == "retriever"
    assert len(workflow.agent_configs) == 2
    assert "calculator" in workflow.agent_configs
    assert "retriever" in workflow.agent_configs


@pytest.mark.asyncio()
async def test_workflow_requires_root_agent():
    """Test that workflow requires exactly one root agent."""
    with pytest.raises(ValueError, match="Exactly one root agent must be provided"):
        MultiAgentWorkflow(
            agent_configs=[
                AgentConfig(
                    name="agent1",
                    description="test",
                    is_entrypoint_agent=True,
                ),
                AgentConfig(
                    name="agent2",
                    description="test",
                    is_entrypoint_agent=True,
                ),
            ]
        )


@pytest.mark.asyncio()
async def test_workflow_execution(calculator_agent, retriever_agent):
    """Test basic workflow execution with agent handoff."""
    workflow = MultiAgentWorkflow(
        agent_configs=[calculator_agent, retriever_agent],
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = workflow.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler

    # Verify we got events indicating handoff and calculation
    assert any(
        ev.current_agent == "retriever" if hasattr(ev, "current_agent") else False
        for ev in events
    )
    assert any(
        ev.current_agent == "calculator" if hasattr(ev, "current_agent") else False
        for ev in events
    )
    assert "8" in response.response


@pytest.mark.asyncio()
async def test_invalid_handoff():
    """Test handling of invalid agent handoff."""
    agent1 = AgentConfig(
        name="agent1",
        description="test",
        is_entrypoint_agent=True,
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
            is_function_calling=True,
        ),
    )

    workflow = MultiAgentWorkflow(
        agent_configs=[agent1],
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
    agent = AgentConfig(
        name="agent",
        description="test",
        is_entrypoint_agent=True,
        llm=MockLLM(
            responses=[
                ChatMessage(
                    role=MessageRole.ASSISTANT, content="Current state processed"
                )
            ],
            is_function_calling=True,
        ),
    )

    workflow = MultiAgentWorkflow(
        agent_configs=[agent],
        initial_state={"counter": 0},
        state_prompt="Current state: {state}. User message: {msg}",
    )

    handler = workflow.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None
