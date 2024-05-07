from typing import Any, AsyncGenerator, Generator, List, Sequence
from unittest.mock import MagicMock, patch

import pytest
from llama_index.agent.openai.base import OpenAIAgent
from llama_index.agent.openai.step import (
    call_tool_with_error_handling,
    advanced_tool_call_parser,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.llms.openai import OpenAI

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def mock_chat_completion(*args: Any, **kwargs: Any) -> ChatCompletion:
    if "functions" in kwargs:
        if not kwargs["functions"]:
            raise ValueError("functions must not be empty")

    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
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
        model="gpt-3.5-turbo-0301",
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


async def mock_achat_completion(*args: Any, **kwargs: Any) -> ChatCompletion:
    return mock_chat_completion(*args, **kwargs)


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


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_basic(MockSyncOpenAI: MagicMock, add_tool: FunctionTool) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "What is 1 + 1?"
    assert agent.chat_history[1].content == "\n\nThis is a test!"


@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.asyncio()
async def test_achat_basic(MockAsyncOpenAI: MagicMock, add_tool: FunctionTool) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_achat_completion()

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
    response = await agent.achat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "What is 1 + 1?"
    assert agent.chat_history[1].content == "\n\nThis is a test!"


@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.skip(reason="currently failing when working on an independent project.")
def test_stream_chat_basic(MockSyncOpenAI: MagicMock, add_tool: FunctionTool) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.side_effect = mock_chat_stream

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
    response = agent.stream_chat("What is 1 + 1?")
    assert isinstance(response, StreamingAgentChatResponse)
    # str() strips newline values
    assert str(response) == "This is a test!"
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "What is 1 + 1?"
    assert agent.chat_history[1].content == "This is a test!"


@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.asyncio()
@pytest.mark.skip(reason="currently failing when working on an independent project.")
async def test_astream_chat_basic(
    MockAsyncOpenAI: MagicMock, add_tool: FunctionTool
) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.side_effect = mock_achat_stream

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
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
def test_chat_no_functions(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = OpenAIAgent.from_tools(
        llm=llm,
    )
    response = agent.chat("What is 1 + 1?")
    assert isinstance(response, AgentChatResponse)
    assert response.response == "\n\nThis is a test!"


def test_call_tool_with_error_handling() -> None:
    """Test call tool with error handling."""

    def _add(a: int, b: int) -> int:
        return a + b

    tool = FunctionTool.from_defaults(fn=_add)

    output = call_tool_with_error_handling(
        tool, {"a": 1, "b": 1}, error_message="Error!"
    )
    assert output.content == "2"

    # try error
    output = call_tool_with_error_handling(
        tool, {"a": "1", "b": 1}, error_message="Error!"
    )
    assert output.content == "Error!"


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_call_tool_with_malformed_function_call(
    MockSyncOpenAI: MagicMock,
    echo_tool: FunctionTool,
) -> None:
    """Test add step."""
    mock_instance = MockSyncOpenAI.return_value
    test_result: str = "This is a test"
    function = Function(name="echo", arguments=f'query = "{test_result}"')
    mock_instance.chat.completions.create.return_value = mock_chat_completion_tool_call(
        function=function
    )

    llm = OpenAI(model="gpt-3.5-turbo")
    # sync
    agent = OpenAIAgent.from_tools(
        tools=[echo_tool],
        llm=llm,
    )
    ## NOTE: can only take a single step before finishing,
    # since mocked chat output does not call any tools
    task = agent.create_task(
        f"This happens if tool call is malformed like:\n{function.arguments}"
    )
    step_output = agent.run_step(task.task_id)
    assert (
        str(step_output.output.sources[0])
        == 'Error in calling tool echo: The input json block is malformed:\n```json\nquery = "This is a test"\n```'
    )


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_call_tool_with_malformed_function_call_and_parser(
    MockSyncOpenAI: MagicMock,
    echo_tool: FunctionTool,
) -> None:
    """Test add step."""
    mock_instance = MockSyncOpenAI.return_value
    test_result: str = "This is a test"
    function = Function(name="echo", arguments=f'query = "{test_result}"')
    mock_instance.chat.completions.create.return_value = mock_chat_completion_tool_call(
        function=function
    )

    llm = OpenAI(model="gpt-3.5-turbo")
    # sync
    agent = OpenAIAgent.from_tools(
        tools=[echo_tool],
        llm=llm,
        tool_call_parser=advanced_tool_call_parser,
    )
    ## NOTE: can only take a single step before finishing,
    # since mocked chat output does not call any tools
    task = agent.create_task(
        f"This happens if tool call is malformed like:\n{function.arguments}"
    )
    step_output = agent.run_step(task.task_id)
    assert str(step_output.output.sources[0]) == test_result


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_add_step(
    MockSyncOpenAI: MagicMock,
    add_tool: FunctionTool,
) -> None:
    """Test add step."""
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion()

    llm = OpenAI(model="gpt-3.5-turbo")
    # sync
    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
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
    # agent = OpenAIAgent.from_tools(
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


@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.asyncio()
async def test_async_add_step(
    MockAsyncOpenAI: MagicMock,
    add_tool: FunctionTool,
) -> None:
    mock_instance = MockAsyncOpenAI.return_value

    llm = OpenAI(model="gpt-3.5-turbo")
    # async
    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
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
    agent = OpenAIAgent.from_tools(
        tools=[add_tool],
        llm=llm,
    )
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
