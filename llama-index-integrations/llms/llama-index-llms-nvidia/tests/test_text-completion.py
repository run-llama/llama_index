import os
from typing import Any, Optional, Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import (
    LLMMetadata,
)
from llama_index.llms.nvidia import NVIDIA as Interface
from llama_index.llms.nvidia.utils import COMPLETION_MODEL_TABLE
from openai.types.completion import Completion, CompletionChoice, CompletionUsage


class CachedNVIDIApiKeys:
    def __init__(self, set_env_key_to: Optional[str] = "", set_fake_key: bool = False):
        self.set_env_key_to = set_env_key_to
        self.set_fake_key = set_fake_key

    def __enter__(self) -> None:
        self.api_env_was = os.environ.get("NVIDIA_API_KEY", "")
        os.environ["NVIDIA_API_KEY"] = self.set_env_key_to

        if self.set_fake_key:
            os.environ["NVIDIA_API_KEY"] = "nvai-" + "x" * 9 + "-" + "x" * 54

    def __exit__(self, *exc: object) -> None:
        if self.api_env_was == "":
            del os.environ["NVIDIA_API_KEY"]
        else:
            os.environ["NVIDIA_API_KEY"] = self.api_env_was


def mock_completion_v1(*args, **kwargs) -> Completion:
    model_name = kwargs.get("model")
    return Completion(
        id="cmpl-4162e407-e121-42b4-8590-1c173380be7d",
        object="text_completion",
        created=1713474384,
        model=model_name,
        usage=CompletionUsage(
            completion_tokens=304, prompt_tokens=11, total_tokens=315
        ),
        choices=[
            CompletionChoice(
                finish_reason="stop", index=0, text="Cool Test Message", logprobs=None
            )
        ],
    )


async def mock_async_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return mock_completion_v1(*args, **kwargs)


def mock_completion_stream_v1(*args, **kwargs) -> Generator[Completion, None, None]:
    model_name = kwargs.get("model")
    responses = [
        Completion(
            id="chatcmpl-998d9b96-0b71-41f5-b910-dd3bc00f38c6",
            object="text_completion",
            created=1713474736,
            model=model_name,
            choices=[CompletionChoice(text="Test", finish_reason="stop", index=0)],
        ),
        Completion(
            id="chatcmpl-998d9b96-0b71-41f5-b910-dd3bc00f38c6",
            object="text_completion",
            created=1713474736,
            model="google/gemma-7b",
            choices=[
                CompletionChoice(text="Second Test", finish_reason="stop", index=0)
            ],
        ),
    ]

    yield from responses


async def mock_async_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[Completion, None]:
    async def gen() -> AsyncGenerator[Completion, None]:
        for response in mock_completion_stream_v1(*args, **kwargs):
            yield response

    return gen()


@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.parametrize("model", COMPLETION_MODEL_TABLE.keys())
def test_model_completions(MockSyncOpenAI: MagicMock, model: str) -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_v1(model=model)

        llm = Interface(model=model)
        prompt = "test prompt"

        response = llm.complete(prompt)
        assert response.text == "Cool Test Message"


def test_validates_api_key_is_present() -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        assert Interface()

        os.environ["NVIDIA_API_KEY"] = ""

        assert Interface(api_key="nvai-" + "x" * 9 + "-" + "x" * 54)


def test_metadata() -> None:
    assert isinstance(Interface().metadata, LLMMetadata)


@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.parametrize("model", COMPLETION_MODEL_TABLE.keys())
def test_completions_model_streaming(MockSyncOpenAI: MagicMock, model: str) -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_stream_v1(
            model=model
        )

        llm = Interface(model=model)
        prompt = "test prompt"

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "TestSecond Test"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.parametrize("model", COMPLETION_MODEL_TABLE.keys())
async def test_async_model_completions(MockAsyncOpenAI: MagicMock, model: str) -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        mock_instance = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(call_args=(model,))
        create_fn.side_effect = mock_async_completion_v1
        mock_instance.completions.create = create_fn

        llm = Interface(model=model)
        prompt = "test prompt"

        response = await llm.acomplete(prompt)
        assert response.text == "Cool Test Message"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.parametrize("model", COMPLETION_MODEL_TABLE.keys())
async def test_async_streaming_completion_model(
    MockAsyncOpenAI: MagicMock, model: str
) -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        mock_instance = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(call_args=(model,))
        create_fn.side_effect = mock_async_completion_stream_v1
        mock_instance.completions.create = create_fn

        llm = Interface(model=model)
        prompt = "test prompt"

        response_gen = await llm.astream_complete(prompt)
        responses = [response async for response in response_gen]
        assert responses[-1].text == "TestSecond Test"
