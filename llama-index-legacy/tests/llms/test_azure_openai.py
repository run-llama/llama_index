from unittest.mock import MagicMock, patch

import httpx
from llama_index.legacy.llms import AzureOpenAI

from tests.llms.test_openai import mock_chat_completion_v1


@patch("llama_index.legacy.llms.azure_openai.SyncAzureOpenAI")
def test_custom_http_client(sync_azure_openai_mock: MagicMock) -> None:
    """
    Verify that a custom http_client set for AzureOpenAI.
    Should get passed on to the implementation from OpenAI.
    """
    custom_http_client = httpx.Client()
    mock_instance = sync_azure_openai_mock.return_value
    # Valid mocked result required to not run into another error
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()
    azure_openai = AzureOpenAI(engine="foo bar", http_client=custom_http_client)
    azure_openai.complete("test prompt")
    sync_azure_openai_mock.assert_called()
    kwargs = sync_azure_openai_mock.call_args.kwargs
    assert "http_client" in kwargs
    assert kwargs["http_client"] == custom_http_client
