from unittest.mock import MagicMock, patch

import httpx
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


@patch("llama_index.embeddings.azure_openai.base.AzureOpenAI")
def test_custom_http_client(azure_openai_mock: MagicMock) -> None:
    """
    Verify that a custom http_client set for AzureOpenAIEmbedding.
    Should get passed on to the implementation from OpenAI.
    """
    custom_http_client = httpx.Client()
    embedding = AzureOpenAIEmbedding(http_client=custom_http_client, api_key="mock")
    embedding._get_client()
    azure_openai_mock.assert_called()
    kwargs = azure_openai_mock.call_args.kwargs
    assert "http_client" in kwargs
    assert kwargs["http_client"] == custom_http_client
