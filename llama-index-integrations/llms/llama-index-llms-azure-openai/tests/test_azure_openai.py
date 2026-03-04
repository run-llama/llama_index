import pytest
from openai import AzureOpenAI as SyncAzureOpenAI
from openai import AsyncAzureOpenAI
from typing import Any, Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.base.llms.types import ChatMessage
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice


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


@patch("llama_index.llms.azure_openai.base.SyncAzureOpenAI")
def test_custom_http_client(sync_azure_openai_mock: MagicMock) -> None:
    """
    Verify that a custom http_client set for AzureOpenAI.
    Should get passed on to the implementation from OpenAI.
    """
    custom_http_client = httpx.Client()
    mock_instance = sync_azure_openai_mock.return_value
    # Valid mocked result required to not run into another error
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()
    azure_openai = AzureOpenAI(
        engine="foo bar", http_client=custom_http_client, api_key="mock"
    )
    azure_openai.complete("test prompt")
    sync_azure_openai_mock.assert_called()
    kwargs = sync_azure_openai_mock.call_args.kwargs
    assert "http_client" in kwargs
    assert kwargs["http_client"] == custom_http_client


@patch("llama_index.llms.azure_openai.base.SyncAzureOpenAI")
def test_custom_azure_ad_token_provider(sync_azure_openai_mock: MagicMock):
    """
    Verify that a custom azure ad token provider set for AzureOpenAI.
    """

    def custom_azure_ad_token_provider() -> str:
        return "mock_api_key"

    mock_instance = sync_azure_openai_mock.return_value
    # Valid mocked result required to not run into another error
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()
    azure_openai = AzureOpenAI(
        engine="foo bar",
        use_azure_ad=True,
        azure_ad_token_provider=custom_azure_ad_token_provider,
    )
    azure_openai.complete("test prompt")
    assert azure_openai.api_key == "mock_api_key"


def mock_chat_completion_stream_with_filter_results(
    *args: Any, **kwargs: Any
) -> Generator[ChatCompletionChunk, None, None]:
    """
    Azure sends a chunk without text content (empty `choices` attribute) as the first chunk.
    It only contains prompt filter results. Documentation on this can be found here: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter?tabs=warning%2Cuser-prompt%2Cpython-new#sample-response-stream-passes-filters.
    """
    responses = [
        ChatCompletionChunk.model_construct(
            id="",
            object="",
            created=0,
            model="",
            prompt_filter_results=[
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
            choices=[],
            usage=None,
        ),
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
                    delta=ChoiceDelta(content="Hello from\n"),
                    finish_reason=None,
                    index=0,
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
                    delta=ChoiceDelta(content="Azure"), finish_reason=None, index=0
                )
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


async def mock_async_chat_completion_stream_with_filter_results(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[ChatCompletionChunk, None]:
    async def gen() -> AsyncGenerator[ChatCompletionChunk, None]:
        for response in mock_chat_completion_stream_with_filter_results(
            *args, **kwargs
        ):
            yield response

    return gen()


@patch("llama_index.llms.azure_openai.base.SyncAzureOpenAI")
def test_chat_completion_with_filter_results(sync_azure_openai_mock: MagicMock) -> None:
    """
    Tests that synchronous chat completions work correctly if first chunk contains prompt
    filter results (empty `choices` list).
    """
    mock_instance = MagicMock(spec=SyncAzureOpenAI)
    sync_azure_openai_mock.return_value = mock_instance

    chat_mock = MagicMock()
    chat_mock.completions.create.return_value = (
        mock_chat_completion_stream_with_filter_results()
    )
    mock_instance.chat = chat_mock

    llm = AzureOpenAI(engine="foo bar", api_key="mock")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response_gen = llm.stream_complete(prompt)
    responses = list(response_gen)
    assert responses[-1].text == "Hello from\nAzure"

    mock_instance.chat.completions.create.return_value = (
        mock_chat_completion_stream_with_filter_results()
    )
    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert chat_responses[-1].message.content == "Hello from\nAzure"
    assert chat_responses[-1].message.role == "assistant"


@pytest.mark.asyncio
@patch("llama_index.llms.azure_openai.base.AsyncAzureOpenAI")
async def test_async_chat_completion_with_filter_results(
    async_azure_openai_mock: MagicMock,
) -> None:
    """
    Tests that asynchronous chat completions work correctly if first chunk contains prompt
    filter results (empty `choices` list).
    """
    mock_instance = MagicMock(spec=AsyncAzureOpenAI)
    async_azure_openai_mock.return_value = mock_instance
    create_fn = AsyncMock()
    create_fn.side_effect = mock_async_chat_completion_stream_with_filter_results
    chat_mock = MagicMock()
    chat_mock.completions.create = create_fn
    mock_instance.chat = chat_mock

    llm = AzureOpenAI(engine="foo bar", api_key="mock")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response_gen = await llm.astream_complete(prompt)
    responses = [item async for item in response_gen]
    assert responses[-1].text == "Hello from\nAzure"

    chat_response_gen = await llm.astream_chat([message])
    chat_responses = [item async for item in chat_response_gen]
    assert chat_responses[-1].message.content == "Hello from\nAzure"
