import re
import time
from typing import Any, List, Sequence

import pytest
from llama_index.core.agent.react.base import ReActAgent, ReActAgentWorker
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.agent.types import Task
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import BaseTool


@pytest.fixture()
def add_tool() -> FunctionTool:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    return FunctionTool.from_defaults(fn=add)


class MockChatLLM(MockLLM):
    _i: int = PrivateAttr()
    _responses: List[ChatMessage] = PrivateAttr()

    def __init__(self, responses: List[ChatMessage]) -> None:
        super().__init__()
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
    _i: int = PrivateAttr()
    _responses: List[ChatMessage] = PrivateAttr()

    def __init__(self, responses: List[ChatMessage]) -> None:
        super().__init__()
        self._i = 0  # call counter, determines which response to return
        self._responses = responses  # list of responses to return

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        del messages  # unused
        full_message = self._responses[self._i]
        self._i += 1

        role = full_message.role
        full_text = full_message.content or ""

        text_so_far = ""
        # create mock stream
        mock_stream = re.split(r"(\s+)", full_text)
        for token in mock_stream:
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

    # exhaust stream
    for delta in response.response_gen:
        continue
    expected_answer = MOCK_STREAM_FINAL_RESPONSE.split("Answer: ")[-1].strip()
    assert response.response == expected_answer

    # there is a very small race condition here that the chat history is not updated
    time.sleep(0.01)

    assert agent.chat_history == [
        ChatMessage(
            content="What is 1 + 1?",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content="2 is the final answer.",
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

    # exhaust stream
    async for delta in response.async_response_gen():
        continue
    expected_answer = MOCK_STREAM_FINAL_RESPONSE.split("Answer: ")[-1].strip()
    assert response.response == expected_answer

    assert agent.chat_history == [
        ChatMessage(
            content="What is 1 + 1?",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content="2 is the final answer.",
            role=MessageRole.ASSISTANT,
        ),
    ]


def _get_agent(
    tools: List[BaseTool],
    streaming: bool = False,
) -> ReActAgent:
    if streaming:
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
    else:
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
    return ReActAgent.from_tools(
        tools=tools,
        llm=mock_llm,
    )


def _get_observations(task: Task) -> List[str]:
    obs_steps = [
        s
        for s in task.extra_state["current_reasoning"]
        if isinstance(s, ObservationReasoningStep)
    ]
    return [s.observation for s in obs_steps]


def test_complaint_when_no_reasoning_step():
    runner = ReActAgent.from_tools(
        tools=[],
        llm=MockLLM(),
    )
    task = runner.create_task("lorem")
    chat_response = ChatResponse(
        message=ChatMessage(
            content="Thought: ipsum\nAction: dolor", role=MessageRole.ASSISTANT
        )
    )
    current_reasoning, is_done = runner.agent_worker._process_actions(
        task, tools=[], output=chat_response
    )
    assert (
        current_reasoning[0].get_content()
        == "Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again."
    )


def test_max_iterations(add_tool: FunctionTool) -> None:
    """Test that _get_response raises ValueError when max_iterations is reached."""
    # Create a minimal mock LLM
    mock_llm = MockLLM()

    # Create agent with a small max_iterations value
    max_iterations = 3
    agent_worker = ReActAgentWorker.from_tools(
        tools=[add_tool],
        llm=mock_llm,
        max_iterations=max_iterations,
    )

    # Create a list of reasoning steps that exceeds max_iterations
    current_reasoning = []
    for i in range(max_iterations + 1):  # Creating more steps than max_iterations
        # Alternate between action and observation steps to simulate a real sequence
        if i % 2 == 0:
            current_reasoning.append(
                ActionReasoningStep(
                    thought=f"Thought {i}", action="add", action_input={"a": i, "b": i}
                )
            )
        else:
            current_reasoning.append(
                ObservationReasoningStep(observation=f"Result: {i + i}")
            )

    # Mock sources
    sources: List[ToolOutput] = []
    # Assert that ValueError is raised with the expected message
    with pytest.raises(ValueError, match="Reached max iterations."):
        agent_worker._get_response(current_reasoning, sources)


def test_add_step(
    add_tool: FunctionTool,
) -> None:
    # sync
    agent = _get_agent([add_tool])
    task = agent.create_task("What is 1 + 1?")
    # first step
    _step_output = agent.run_step(task.task_id)
    # add human input (not used but should be in memory)
    _step_output = agent.run_step(task.task_id, input="tmp")
    observations = _get_observations(task)
    assert "tmp" in observations

    # stream_step
    agent = _get_agent([add_tool], streaming=True)
    task = agent.create_task("What is 1 + 1?")
    # first step
    _step_output = agent.stream_step(task.task_id)
    # add human input (not used but should be in memory)
    _step_output = agent.stream_step(task.task_id, input="tmp")
    observations = _get_observations(task)
    assert "tmp" in observations


@pytest.mark.asyncio
async def test_async_add_step(
    add_tool: FunctionTool,
) -> None:
    # async
    agent = _get_agent([add_tool])
    task = agent.create_task("What is 1 + 1?")
    # first step
    _step_output = await agent.arun_step(task.task_id)
    # add human input (not used but should be in memory)
    _step_output = await agent.arun_step(task.task_id, input="tmp")
    observations = _get_observations(task)
    assert "tmp" in observations

    # async stream step
    agent = _get_agent([add_tool], streaming=True)
    task = agent.create_task("What is 1 + 1?")
    # first step
    _step_output = await agent.astream_step(task.task_id)
    # add human input (not used but should be in memory)
    _step_output = await agent.astream_step(task.task_id, input="tmp")
    observations = _get_observations(task)
    assert "tmp" in observations
