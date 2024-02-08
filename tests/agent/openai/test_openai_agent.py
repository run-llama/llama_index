from typing import Any, AsyncGenerator, Generator, List, Sequence
from unittest.mock import MagicMock, patch

import pytest
from llama_index.agent.openai.base import OpenAIAgent
from llama_index.agent.openai.step import call_tool_with_error_handling
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.core.llms.types import ChatMessage, ChatResponse
from llama_index.llms.base import ChatMessage, ChatResponse
from llama_index.llms.mock import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.tools.function_tool import FunctionTool
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage


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
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    content="\n\nThis is a test!",
                ),
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
            usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
            choices=[
                Choice(
                    message=ChatCompletionMessage(
                        role="assistant", content="\n\nThis is a test!"
                    ),
                    finish_reason="stop",
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="\n\nThis is a test!",
                    ),
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


@patch("llama_index.llms.openai.SyncOpenAI")
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


@patch("llama_index.llms.openai.AsyncOpenAI")
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


@patch("llama_index.llms.openai.SyncOpenAI")
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


@patch("llama_index.llms.openai.AsyncOpenAI")
@pytest.mark.asyncio()
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


@patch("llama_index.llms.openai.SyncOpenAI")
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


@patch("llama_index.llms.openai.SyncOpenAI")
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


@patch("llama_index.llms.openai.AsyncOpenAI")
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
    step_output = await agent.astream_step(task.task_id, input="tmp")
    chat_history = task.extra_state["new_memory"].get_all()
    assert "tmp" in [m.content for m in chat_history]
