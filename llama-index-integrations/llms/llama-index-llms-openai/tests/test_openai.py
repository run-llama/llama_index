import os
from typing import Any, AsyncGenerator, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ThinkingBlock, TextBlock
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import O1_MODELS

import openai
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion import Completion, CompletionChoice, CompletionUsage


class CachedOpenAIApiKeys:
    """
    Saves the users' OpenAI API key and OpenAI API type either in
    the environment variable or set to the library itself.
    This allows us to run tests by setting it without plowing over
    the local environment.
    """

    def __init__(
        self,
        set_env_key_to: Optional[str] = "",
        set_library_key_to: Optional[str] = None,
        set_fake_key: bool = False,
        set_env_type_to: Optional[str] = "",
        set_library_type_to: str = "open_ai",  # default value in openai package
    ):
        self.set_env_key_to = set_env_key_to
        self.set_library_key_to = set_library_key_to
        self.set_fake_key = set_fake_key
        self.set_env_type_to = set_env_type_to
        self.set_library_type_to = set_library_type_to

    def __enter__(self) -> None:
        self.api_env_variable_was = os.environ.get("OPENAI_API_KEY", "")
        self.api_env_type_was = os.environ.get("OPENAI_API_TYPE", "")
        self.openai_api_key_was = openai.api_key
        self.openai_api_type_was = openai.api_type

        os.environ["OPENAI_API_KEY"] = str(self.set_env_key_to)
        os.environ["OPENAI_API_TYPE"] = str(self.set_env_type_to)

        if self.set_fake_key:
            os.environ["OPENAI_API_KEY"] = "sk-" + "a" * 48

    # No matter what, set the environment variable back to what it was
    def __exit__(self, *exc: object) -> None:
        os.environ["OPENAI_API_KEY"] = str(self.api_env_variable_was)
        os.environ["OPENAI_API_TYPE"] = str(self.api_env_type_was)
        openai.api_key = self.openai_api_key_was
        openai.api_type = self.openai_api_type_was


def mock_completion(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://platform.openai.com/docs/api-reference/completions/create
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def mock_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return Completion(
        id="cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        object="text_completion",
        created=1589478378,
        model="text-davinci-003",
        choices=[
            CompletionChoice(
                text="\n\nThis is indeed a test",
                index=0,
                logprobs=None,
                finish_reason="length",
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


async def mock_async_completion(*args: Any, **kwargs: Any) -> dict:
    return mock_completion(*args, **kwargs)


async def mock_async_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return mock_completion_v1(*args, **kwargs)


def mock_chat_completion(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "\n\nThis is a test!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def mock_chat_completion_v1(*args: Any, **kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
        usage=CompletionUsage(prompt_tokens=13, completion_tokens=7, total_tokens=20),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def mock_completion_stream(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
    # Example taken from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    responses = [
        {
            "choices": [
                {
                    "text": "1",
                }
            ],
        },
        {
            "choices": [
                {
                    "text": "2",
                }
            ],
        },
    ]
    yield from responses


def mock_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> Generator[Completion, None, None]:
    responses = [
        Completion(
            id="cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            object="text_completion",
            created=1589478378,
            model="text-davinci-003",
            choices=[CompletionChoice(text="1", finish_reason="stop", index=0)],
        ),
        Completion(
            id="cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            object="text_completion",
            created=1589478378,
            model="text-davinci-003",
            choices=[CompletionChoice(text="2", finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


async def mock_async_completion_stream(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[dict, None]:
    async def gen() -> AsyncGenerator[dict, None]:
        for response in mock_completion_stream(*args, **kwargs):
            yield response

    return gen()


async def mock_async_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[Completion, None]:
    async def gen() -> AsyncGenerator[Completion, None]:
        for response in mock_completion_stream_v1(*args, **kwargs):
            yield response

    return gen()


def mock_chat_completion_stream(
    *args: Any, **kwargs: Any
) -> Generator[dict, None, None]:
    # Example taken from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    responses = [
        {
            "choices": [
                {"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {"delta": {"content": "\n\n"}, "finish_reason": None, "index": 0}
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {"content": "2"}, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
    ]
    yield from responses


def mock_chat_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> Generator[ChatCompletionChunk, None, None]:
    responses = [
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="\n\n"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(delta=ChoiceDelta(content="2"), finish_reason=None, index=0)
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[ChunkChoice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_completion_model_basic(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_v1()

        llm = OpenAI(model="text-davinci-003")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is indeed a test"
        assert response.additional_kwargs["total_tokens"] == 12

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is indeed a test"
        assert chat_response.message.additional_kwargs["total_tokens"] == 12


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_model_basic(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()

        llm = OpenAI(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is a test!"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is a test!"
        assert chat_response.additional_kwargs["total_tokens"] == 20


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_completion_model_streaming(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_stream_v1()

        llm = OpenAI(model="text-davinci-003")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "12"

        mock_instance.completions.create.return_value = mock_completion_stream_v1()
        chat_response_gen = llm.stream_chat([message])
        chat_responses = list(chat_response_gen)
        assert chat_responses[-1].message.content == "12"


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_model_streaming(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream_v1()
        )

        llm = OpenAI(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "\n\n2"

        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream_v1()
        )
        chat_response_gen = llm.stream_chat([message])
        chat_responses = list(chat_response_gen)
        assert chat_responses[-1].message.blocks[-1].text == "\n\n2"
        assert chat_responses[-1].message.role == "assistant"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_completion_model_async(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    create_fn = AsyncMock()
    create_fn.side_effect = mock_async_completion_v1
    mock_instance.completions.create = create_fn

    llm = OpenAI(model="text-davinci-003")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = await llm.acomplete(prompt)
    assert response.text == "\n\nThis is indeed a test"

    chat_response = await llm.achat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_completion_model_async_streaming(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    create_fn = AsyncMock()
    create_fn.side_effect = mock_async_completion_stream_v1
    mock_instance.completions.create = create_fn

    llm = OpenAI(model="text-davinci-003")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response_gen = await llm.astream_complete(prompt)
    responses = [item async for item in response_gen]
    assert responses[-1].text == "12"

    chat_response_gen = await llm.astream_chat([message])
    chat_responses = [item async for item in chat_response_gen]
    assert chat_responses[-1].message.content == "12"


def test_validates_api_key_is_present() -> None:
    with CachedOpenAIApiKeys():
        os.environ["OPENAI_API_KEY"] = "sk-" + ("a" * 48)

        # We can create a new LLM when the env variable is set
        assert OpenAI()

        os.environ["OPENAI_API_KEY"] = ""

        # We can create a new LLM when the api_key is set on the
        # class directly
        assert OpenAI(api_key="sk-" + ("a" * 48))


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_completion_model_with_retry(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.completions.create.side_effect = openai.APITimeoutError(None)

    llm = OpenAI(model="text-davinci-003", max_retries=3)
    prompt = "test prompt"
    with pytest.raises(openai.APITimeoutError) as exc:
        llm.complete(prompt)

    assert exc.value.message == "Request timed out."
    # The actual retry count is max_retries - 1
    # see https://github.com/jd/tenacity/issues/459
    assert mock_instance.completions.create.call_count == 3


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_ensure_chat_message_is_serializable(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()

        llm = OpenAI(model="gpt-3.5-turbo")
        message = ChatMessage(role="user", content="test message")

        response = llm.chat([message])
        response.message.additional_kwargs["test"] = ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant", content="test"),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        data = response.message.dict()
        assert isinstance(data, dict)
        assert isinstance(data["additional_kwargs"], dict)
        assert isinstance(data["additional_kwargs"]["test"]["choices"], list)
        assert (
            data["additional_kwargs"]["test"]["choices"][0]["delta"]["content"]
            == "test"
        )


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_structured_chat_simple(MockSyncOpenAI: MagicMock):
    """Simple test for structured output using as_structured_llm."""
    from pydantic import BaseModel, Field
    from llama_index.core.base.llms.types import ChatMessage

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    # Mock OpenAI response structure
    mock_choice = MagicMock()
    mock_choice.message.role = "assistant"
    mock_choice.message.content = '{"name": "Alice", "age": 25}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Mock OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    MockSyncOpenAI.return_value = mock_client

    llm = OpenAI(model="gpt-4o", api_key="test-key")
    structured_llm = llm.as_structured_llm(Person)
    messages = [
        ChatMessage(role="user", content="Create a person named Alice who is 25")
    ]

    result = structured_llm.chat(messages)
    # Verify the result has the expected structure
    assert isinstance(result.raw, Person)


def test_prepare_schema_sanitizes_json_schema_name() -> None:
    from pydantic import BaseModel

    class DummyModel(BaseModel):
        answer: int

    llm = OpenAI(model="gpt-4o", api_key="test-key")
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "GenericDataModel[int]", "schema": {}},
    }

    with patch(
        "openai.resources.chat.completions.completions._type_to_response_format",
        return_value=response_format,
    ):
        llm_kwargs = llm._prepare_schema({}, DummyModel)

    assert (
        llm_kwargs["response_format"]["json_schema"]["name"] == "GenericDataModel_int_"
    )


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_structured_chat_simple_async(MockAsyncOpenAI: MagicMock):
    """Simple async test for structured output using as_structured_llm."""
    from pydantic import BaseModel, Field
    from llama_index.core.base.llms.types import ChatMessage

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    # Mock OpenAI response structure
    mock_choice = MagicMock()
    mock_choice.message.role = "assistant"
    mock_choice.message.content = '{"name": "Bob", "age": 30}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Mock async OpenAI client
    mock_client = MagicMock()
    create_fn = AsyncMock()
    create_fn.return_value = mock_response
    mock_client.chat.completions.create = create_fn
    MockAsyncOpenAI.return_value = mock_client

    # Instantiate OpenAI class
    llm = OpenAI(model="gpt-4o", api_key="test-key")
    structured_llm = llm.as_structured_llm(Person)
    messages = [ChatMessage(role="user", content="Create a person named Bob who is 30")]
    result = await structured_llm.achat(messages)

    # Verify the result has the expected structure
    assert isinstance(result.raw, Person)


@pytest.mark.parametrize(
    "effort", ["low", "medium", "high", "minimal", "xhigh", "none"]
)
def test_reasoning_effort_passed_for_o1_models(effort):
    """Test that reasoning_effort is passed for O1 models."""
    model_name = "o1-mini"
    assert model_name in O1_MODELS

    llm = OpenAI(model=model_name, reasoning_effort=effort, api_key="test-key")
    kwargs = llm._get_model_kwargs()
    assert "reasoning_effort" in kwargs
    assert kwargs["reasoning_effort"] == effort


def test_reasoning_effort_not_passed_for_non_o1_models():
    """Test that reasoning_effort is NOT passed for non-O1 models."""
    model_name = "gpt-4o"
    assert model_name not in O1_MODELS

    llm = OpenAI(model=model_name, reasoning_effort="low", api_key="test-key")
    kwargs = llm._get_model_kwargs()
    assert "reasoning_effort" not in kwargs


def test_reasoning_effort_none_default():
    """Test that reasoning_effort defaults to None and is not passed."""
    model_name = "o1-mini"

    llm = OpenAI(model=model_name, api_key="test-key")
    kwargs = llm._get_model_kwargs()
    assert "reasoning_effort" not in kwargs


# ===== reasoning_content tests (OpenAI-compatible providers) =====


def _make_chunk(
    delta_kwargs: dict, finish_reason: Optional[str] = None
) -> ChatCompletionChunk:
    """Helper to create a single ChatCompletionChunk."""
    extra = delta_kwargs.pop("__extra__", None)
    chunk = ChatCompletionChunk(
        id="chatcmpl-reasoning",
        object="chat.completion.chunk",
        created=1700000000,
        model="qwen3-thinking",
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(**delta_kwargs),
                finish_reason=finish_reason,
                index=0,
            )
        ],
    )
    if extra:
        chunk.choices[0].delta.__pydantic_extra__ = extra
    return chunk


def _make_reasoning_stream_chunks() -> list[ChatCompletionChunk]:
    """Simulate an OpenAI-compatible API streaming reasoning_content then content."""
    return [
        _make_chunk({"role": "assistant"}),
        _make_chunk(
            {"content": None, "__extra__": {"reasoning_content": "Let me think"}}
        ),
        _make_chunk(
            {"content": None, "__extra__": {"reasoning_content": " about this."}}
        ),
        _make_chunk({"content": "The answer"}),
        _make_chunk({"content": " is 42."}),
        _make_chunk({}, finish_reason="stop"),
    ]


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_stream_chat_reasoning_content(MockSyncOpenAI: MagicMock) -> None:
    """Test that reasoning_content from streaming is captured as ThinkingBlock and thinking_delta."""
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = iter(
            _make_reasoning_stream_chunks()
        )

        llm = OpenAI(model="gpt-4o", api_key="test-key")
        responses = list(llm.stream_chat([ChatMessage(role="user", content="test")]))

        final = responses[-1]
        thinking_blocks = [
            b for b in final.message.blocks if isinstance(b, ThinkingBlock)
        ]
        text_blocks = [b for b in final.message.blocks if isinstance(b, TextBlock)]

        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].content == "Let me think about this."
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "The answer is 42."

        # Exactly 2 chunks carry thinking_delta (the two reasoning chunks)
        reasoning_chunks = [
            r for r in responses if r.additional_kwargs.get("thinking_delta")
        ]
        assert len(reasoning_chunks) == 2
        assert reasoning_chunks[0].additional_kwargs["thinking_delta"] == "Let me think"
        assert reasoning_chunks[1].additional_kwargs["thinking_delta"] == " about this."


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_astream_chat_reasoning_content(MockAsyncOpenAI: MagicMock) -> None:
    """Test that reasoning_content from async streaming is captured as ThinkingBlock."""
    mock_instance = MockAsyncOpenAI.return_value

    async def mock_async_stream(*args: Any, **kwargs: Any) -> AsyncGenerator:
        for chunk in _make_reasoning_stream_chunks():
            yield chunk

    create_fn = AsyncMock()
    create_fn.return_value = mock_async_stream()
    mock_instance.chat.completions.create = create_fn

    llm = OpenAI(model="gpt-4o", api_key="test-key")
    response_gen = await llm.astream_chat([ChatMessage(role="user", content="test")])
    responses = [r async for r in response_gen]

    final = responses[-1]
    thinking_blocks = [b for b in final.message.blocks if isinstance(b, ThinkingBlock)]
    text_blocks = [b for b in final.message.blocks if isinstance(b, TextBlock)]

    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].content == "Let me think about this."
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "The answer is 42."

    # Verify thinking_delta on async path too
    reasoning_chunks = [
        r for r in responses if r.additional_kwargs.get("thinking_delta")
    ]
    assert len(reasoning_chunks) == 2


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_reasoning_content_non_streaming(MockSyncOpenAI: MagicMock) -> None:
    """Test that reasoning_content in non-streaming responses is captured as ThinkingBlock."""
    with CachedOpenAIApiKeys(set_fake_key=True):
        response = ChatCompletion(
            id="chatcmpl-reasoning",
            object="chat.completion",
            created=1700000000,
            model="qwen3-thinking",
            choices=[
                Choice(
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="The answer is 42.",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
        )
        response.choices[0].message.__pydantic_extra__ = {
            "reasoning_content": "Let me think step by step..."
        }

        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = response

        llm = OpenAI(model="gpt-4o", api_key="test-key")
        result = llm.chat([ChatMessage(role="user", content="test")])

        thinking_blocks = [
            b for b in result.message.blocks if isinstance(b, ThinkingBlock)
        ]
        text_blocks = [b for b in result.message.blocks if isinstance(b, TextBlock)]

        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].content == "Let me think step by step..."
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "The answer is 42."


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_stream_chat_no_reasoning_content(MockSyncOpenAI: MagicMock) -> None:
    """Test that streaming without reasoning_content produces no ThinkingBlock."""
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream_v1()
        )

        llm = OpenAI(model="gpt-4o", api_key="test-key")
        responses = list(llm.stream_chat([ChatMessage(role="user", content="test")]))

        final = responses[-1]
        thinking_blocks = [
            b for b in final.message.blocks if isinstance(b, ThinkingBlock)
        ]
        assert len(thinking_blocks) == 0
        assert final.message.content == "\n\n2"


def test_to_openai_message_dict_skips_thinking_block() -> None:
    """Test that ThinkingBlock is skipped when converting messages to OpenAI format."""
    from llama_index.llms.openai.utils import to_openai_message_dict

    message = ChatMessage(
        role="assistant",
        blocks=[
            ThinkingBlock(content="internal reasoning"),
            TextBlock(text="The answer is 42."),
        ],
    )

    result = to_openai_message_dict(message)
    assert result["role"] == "assistant"
    assert result["content"] == "The answer is 42."


def test_from_openai_message_with_reasoning_content() -> None:
    """Test that from_openai_message extracts reasoning_content as ThinkingBlock."""
    from llama_index.llms.openai.utils import from_openai_message

    openai_msg = ChatCompletionMessage(
        role="assistant",
        content="The answer is 42.",
    )
    openai_msg.__pydantic_extra__ = {"reasoning_content": "Let me think..."}

    result = from_openai_message(openai_msg, modalities=["text"])

    thinking_blocks = [b for b in result.blocks if isinstance(b, ThinkingBlock)]
    text_blocks = [b for b in result.blocks if isinstance(b, TextBlock)]

    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].content == "Let me think..."
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "The answer is 42."


def test_from_openai_message_without_reasoning_content() -> None:
    """Test that from_openai_message works normally without reasoning_content."""
    from llama_index.llms.openai.utils import from_openai_message

    openai_msg = ChatCompletionMessage(
        role="assistant",
        content="Hello!",
    )

    result = from_openai_message(openai_msg, modalities=["text"])

    thinking_blocks = [b for b in result.blocks if isinstance(b, ThinkingBlock)]
    assert len(thinking_blocks) == 0
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "Hello!"
