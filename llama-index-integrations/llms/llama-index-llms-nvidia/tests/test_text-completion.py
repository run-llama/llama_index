import os
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.nvidia import NVIDIA as Interface
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


def mock_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return Completion(
        id="cmpl-4162e407-e121-42b4-8590-1c173380be7d",
        object="text_completion",
        created=1713474384,
        model="mistralai/mistral-7b-instruct-v0.2",
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


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_model_basic(MockSyncOpenAI: MagicMock) -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_v1()

        llm = Interface(model="bigcode/starcoder2-7b")
        prompt = "test prompt"

        response = llm.complete(prompt)
        assert response.text == "Cool Test Message"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_async_model_basic(MockAsyncOpenAI: MagicMock) -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        mock_instance = MockAsyncOpenAI.return_value
        create_fn = AsyncMock()
        create_fn.side_effect = mock_async_completion_v1
        mock_instance.completions.create = create_fn

        llm = Interface(model="bigcode/starcoder2-7b")
        prompt = "test prompt"

        response = await llm.acomplete(prompt)
        assert response.text == "Cool Test Message"


def test_validates_api_key_is_present() -> None:
    with CachedNVIDIApiKeys(set_fake_key=True):
        assert Interface()

        os.environ["NVIDIA_API_KEY"] = ""

        assert Interface(api_key="nvai-" + "x" * 9 + "-" + "x" * 54)


def test_metadata() -> None:
    assert isinstance(Interface().metadata, LLMMetadata)
