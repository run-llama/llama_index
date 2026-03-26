import json
from typing import (
    Any,
    AsyncGenerator,
    Generator,
)
from unittest.mock import MagicMock, patch
import pytest
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
from llama_index.core.agent.workflow import FunctionAgent


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


@pytest.mark.asyncio
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_run_basic(
    MockAsyncOpenAI: MagicMock, add_tool: FunctionTool, masked_env_var
) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_achat_stream()

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    agent = FunctionAgent(
        tools=[add_tool],
        llm=llm,
    )
    response = await agent.run("What is 1 + 1?")
    assert "\n\nThis is a test!" in str(response)


@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_run_no_functions(MockAsyncOpenAI: MagicMock, masked_env_var) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_achat_stream()

    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    agent = FunctionAgent(
        tools=[],
        llm=llm,
    )
    response = await agent.run("What is 1 + 1?")
    assert "\n\nThis is a test!" in str(response)
