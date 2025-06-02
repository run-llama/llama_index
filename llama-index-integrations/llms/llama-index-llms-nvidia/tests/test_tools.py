import json
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    List,
)
from unittest.mock import MagicMock, patch
from llama_index.core.base.agent.types import TaskStepOutput
import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.llms.nvidia import NVIDIA
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from llama_index.core.agent import FunctionCallingAgentWorker


def mock_chat_completion(*args: Any, **kwargs: Any) -> ChatCompletion:
    if "functions" in kwargs:
        if not kwargs["functions"]:
            raise ValueError("functions must not be empty")

    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="meta/llama-3.1-8b-instruct",
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
                logprobs=None,
            )
        ],
    )


def mock_chat_completion_tool_call(
    function: Function, *args: Any, **kwargs: Any
) -> ChatCompletion:
    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="meta/llama-3.1-8b-instruct",
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content="\n\nThis is a test!",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="toolcall-abc123",
                            function=function,
                            type="function",
                        )
                    ],
                ),
                finish_reason="stop",
                index=0,
                logprobs=None,
            )
        ],
    )


def mock_chat_stream(
    *args: Any, **kwargs: Any
) -> Generator[ChatCompletionChunk, None, None]:
    if "functions" in kwargs:
        if not kwargs["functions"]:
            raise ValueError("functions must not be empty")

    yield ChatCompletionChunk(
        id="chatcmpl-abc123",
        object="chat.completion.chunk",
        created=1677858242,
        model="meta/llama-3.1-8b-instruct",
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(role="assistant", content="\n\nThis is a test!"),
                finish_reason="stop",
                index=0,
                logprobs=None,
            )
        ],
    )


@pytest.mark.asyncio
async def mock_achat_completion(*args: Any, **kwargs: Any) -> ChatCompletion:
    return mock_chat_completion(*args, **kwargs)


@pytest.mark.asyncio
async def mock_achat_completion_tool_call(
    function: Function, *args: Any, **kwargs: Any
) -> ChatCompletion:
    return mock_chat_completion_tool_call(function, *args, **kwargs)


@pytest.mark.asyncio
async def mock_achat_stream(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[ChatCompletionChunk, None]:
    async def _mock_achat_stream(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        if "functions" in kwargs:
            if not kwargs["functions"]:
                raise ValueError("functions must not be empty")

        yield ChatCompletionChunk(
            id="chatcmpl-abc123",
            object="chat.completion.chunk",
            created=1677858242,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant", content="\n\nThis is a test!"),
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                )
            ],
        )

    return _mock_achat_stream(*args, **kwargs)


@pytest.fixture()
def add_tool() -> FunctionTool:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    return FunctionTool.from_defaults(fn=add)


@pytest.fixture()
def echo_tool() -> FunctionTool:
    def echo(query: str) -> str:
        """Echos input."""
        return query

    return FunctionTool.from_defaults(fn=echo)


@pytest.fixture()
def malformed_echo_function() -> Function:
    test_result: str = "This is a test"
    return Function(name="echo", arguments=f'query = "{test_result}"')


@pytest.fixture()
def echo_function() -> Function:
    test_result: str = "This is a test"
    return Function(name="echo", arguments=json.dumps({"query": test_result}))


MOCK_ACTION_RESPONSE = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: {"a": 1, "b": 1}
"""

MOCK_FINAL_RESPONSE = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: 2
"""


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_basic(
    MockSyncOpenAI: MagicMock, add_tool: FunctionTool, masked_env_var
) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    agent = FunctionCallingAgentWorker.from_tools(
        tools=[add_tool],
        llm=llm,
    ).as_agent()
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "What is 1 + 1?"
    assert agent.chat_history[1].content == "\n\nThis is a test!"


@pytest.mark.asyncio
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_achat_basic(
    MockAsyncOpenAI: MagicMock, add_tool: FunctionTool, masked_env_var
) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_achat_completion()

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    agent = FunctionCallingAgentWorker.from_tools(
        tools=[add_tool],
        llm=llm,
    ).as_agent()
    response = await agent.achat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "What is 1 + 1?"
    assert agent.chat_history[1].content == "\n\nThis is a test!"


@pytest.mark.xfail(
    reason="streaming not yet implemented, see https://github.com/run-llama/llama_index/discussions/14653 and https://github.com/run-llama/llama_index/issues/15079"
)
@pytest.mark.asyncio
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_astream_chat_basic(
    MockAsyncOpenAI: MagicMock, add_tool: FunctionTool, masked_env_var
) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.side_effect = mock_achat_stream

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    agent = FunctionCallingAgentWorker.from_tools(
        tools=[add_tool],
        llm=llm,
    ).as_agent()
    response_stream = await agent.astream_chat("What is 1 + 1?")
    async for response in response_stream.async_response_gen():
        pass
    assert isinstance(response_stream, StreamingAgentChatResponse)
    # str() strips newline values
    assert response == "\n\nThis is a test!"
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "What is 1 + 1?"
    assert agent.chat_history[1].content == "This is a test!"


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_no_functions(MockSyncOpenAI: MagicMock, masked_env_var) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    agent = FunctionCallingAgentWorker.from_tools(
        llm=llm,
    ).as_agent()
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_add_step(
    MockSyncOpenAI: MagicMock, add_tool: FunctionTool, masked_env_var
) -> None:
    """Test add step."""
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
    # sync
    agent = FunctionCallingAgentWorker.from_tools(
        tools=[add_tool],
        llm=llm,
    ).as_agent()
    ## NOTE: can only take a single step before finishing,
    # since mocked chat output does not call any tools
    task = agent.create_task("What is 1 + 1?")
    step_output = agent.run_step(task.task_id)
    assert str(step_output) == "\n\nThis is a test!"

    # add human input (not used but should be in memory)
    task = agent.create_task("What is 1 + 1?")
    step_output = agent.run_step(task.task_id, input="tmp")
    chat_history: List[ChatMessage] = task.extra_state["new_memory"].get_all()
    assert "tmp" in [m.content for m in chat_history]

    # # stream_step
    # agent = FunctionCallingAgentWorker.from_tools(
    #     tools=[add_tool],
    #     llm=llm,
    # )
    # task = agent.create_task("What is 1 + 1?")
    # # first step
    # step_output = agent.stream_step(task.task_id)
    # # add human input (not used but should be in memory)
    # step_output = agent.stream_step(task.task_id, input="tmp")
    # chat_history: List[ChatMessage] = task.extra_state["new_memory"].get_all()
    # assert "tmp" in [m.content for m in chat_history]


@pytest.mark.xfail(
    reason="streaming not yet implemented, see https://github.com/run-llama/llama_index/discussions/14653 and https://github.com/run-llama/llama_index/issues/15079"
)
@pytest.mark.asyncio
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_async_add_step(
    MockAsyncOpenAI: MagicMock, add_tool: FunctionTool, masked_env_var
) -> None:
    mock_instance = MockAsyncOpenAI.return_value

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
    # async
    agent = FunctionCallingAgentWorker.from_tools(
        tools=[add_tool],
        llm=llm,
    ).as_agent()
    task = agent.create_task("What is 1 + 1?")
    # first step
    mock_instance.chat.completions.create.return_value = mock_achat_completion()
    step_output = await agent.arun_step(task.task_id)
    # add human input (not used but should be in memory)
    task = agent.create_task("What is 1 + 1?")
    mock_instance.chat.completions.create.return_value = mock_achat_completion()
    step_output = await agent.arun_step(task.task_id, input="tmp")
    chat_history: List[ChatMessage] = task.extra_state["new_memory"].get_all()
    assert "tmp" in [m.content for m in chat_history]

    # async stream step
    agent = FunctionCallingAgentWorker.from_tools(
        tools=[add_tool],
        llm=llm,
    ).as_agent()
    task = agent.create_task("What is 1 + 1?")
    # first step
    mock_instance.chat.completions.create.side_effect = mock_achat_stream
    step_output = await agent.astream_step(task.task_id)
    # add human input (not used but should be in memory)
    task = agent.create_task("What is 1 + 1?")
    mock_instance.chat.completions.create.side_effect = mock_achat_stream

    # stream the output to ensure it gets written to memory
    step_output = await agent.astream_step(task.task_id, input="tmp")
    async for _ in step_output.output.async_response_gen():
        pass

    chat_history = task.memory.get_all()
    assert "tmp" in [m.content for m in chat_history]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run_step", "arun_step"])
@patch("llama_index.llms.openai.base.SyncOpenAI")
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_run_step_returns_correct_sources_history(
    MockAsyncOpenAI: MagicMock,
    MockSyncOpenAI: MagicMock,
    method: str,
    echo_tool: FunctionTool,
    echo_function: Function,
    masked_env_var,
) -> None:
    num_steps = 4
    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
    agent = FunctionCallingAgentWorker.from_tools(
        tools=[echo_tool],
        llm=llm,
    ).as_agent()
    task = agent.create_task("")
    step_outputs: List[TaskStepOutput] = []

    if method == "run_step":
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_tool_call(echo_function)
        )
    else:
        mock_instance = MockAsyncOpenAI.return_value
        mock_instance.chat.completions.create.side_effect = [
            mock_achat_completion_tool_call(echo_function) for _ in range(num_steps)
        ]

    # Create steps
    steps = [agent.agent_worker.initialize_step(task)]
    for step_idx in range(num_steps - 1):
        steps.append(
            steps[-1].get_next_step(
                step_id=str(uuid.uuid4()),
                input=None,
            )
        )

    # Run each step, invoking a single tool call each time
    for step_idx in range(num_steps):
        step_outputs.append(
            agent.agent_worker.run_step(steps[step_idx], task)
            if method == "run_step"
            else await agent.agent_worker.arun_step(steps[step_idx], task)
        )

    # Ensure that each step only has one source for its one tool call
    for step_idx in range(num_steps):
        assert len(step_outputs[step_idx].output.sources) == 1
