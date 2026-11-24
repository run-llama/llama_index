from unittest.mock import MagicMock, patch

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


def _embedding() -> AzureOpenAIEmbedding:
    return AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="test-deployment",
        azure_endpoint="https://example.openai.azure.com/",
        api_key="fake-key",
        api_version="2024-02-01",
        max_retries=3,
        timeout=5.0,
    )


@patch("llama_index.embeddings.azure_openai.base.AzureOpenAI")
def test_sync_client_gets_timeout_and_max_retries(mock_client: MagicMock) -> None:
    _embedding()._get_client()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["timeout"] == 5.0
    assert kwargs["max_retries"] == 3


@patch("llama_index.embeddings.azure_openai.base.AsyncAzureOpenAI")
def test_async_client_gets_timeout_and_max_retries(mock_client: MagicMock) -> None:
    _embedding()._get_aclient()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["timeout"] == 5.0
    assert kwargs["max_retries"] == 3


def test_real_client_uses_timeout_and_max_retries() -> None:
    client = _embedding()._get_client()
    assert client.timeout == 5.0
    assert client.max_retries == 3
