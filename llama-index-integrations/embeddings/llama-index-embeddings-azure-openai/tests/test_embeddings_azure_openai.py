from unittest.mock import MagicMock, patch

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


def _embedding_with_provider():
    def custom_azure_ad_token_provider() -> str:
        return "mock_api_key"

    embedding = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="test-deployment",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-02-01",
        use_azure_ad=True,
        azure_ad_token_provider=custom_azure_ad_token_provider,
    )
    return embedding, custom_azure_ad_token_provider


@patch("llama_index.embeddings.azure_openai.base.AzureOpenAI")
def test_azure_ad_token_provider_sync_client(mock_client: MagicMock) -> None:
    """
    When a token provider is configured, only the provider is passed to
    the SDK: openai>=3.4.0 rejects passing both api_key and
    azure_ad_token_provider (MutuallyExclusiveAuthError).
    """
    embedding, provider = _embedding_with_provider()
    embedding._get_client()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["azure_ad_token_provider"] is provider
    assert "api_key" not in kwargs


@patch("llama_index.embeddings.azure_openai.base.AsyncAzureOpenAI")
def test_azure_ad_token_provider_async_client(mock_client: MagicMock) -> None:
    """
    When a token provider is configured, only the provider is passed to
    the SDK: openai>=3.4.0 rejects passing both api_key and
    azure_ad_token_provider (MutuallyExclusiveAuthError).
    """
    embedding, provider = _embedding_with_provider()
    embedding._get_aclient()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["azure_ad_token_provider"] is provider
    assert "api_key" not in kwargs
