from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from llama_index.llms.azure_openai import AzureOpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import CompletionUsage


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
