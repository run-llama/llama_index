import pytest
from unittest.mock import Mock, patch, AsyncMock
from llama_index.embeddings.nvidia import NVIDIAEmbedding


class MockEmbeddingResponse:
    """Mock response matching the structure expected by the code."""

    def __init__(self):
        self.data = [Mock(embedding=[1.0, 2.0, 3.0], index=0)]


@pytest.fixture(autouse=True)
def mock_openai():
    """Set up mock for OpenAI client."""
    # Create mock response
    mock_response = MockEmbeddingResponse()

    # Patch at the module level where the code imports from
    # NVIDIAEmbedding uses: from openai import OpenAI, AsyncOpenAI
    with (
        patch("llama_index.embeddings.nvidia.base.OpenAI") as mock_openai_cls,
        patch(
            "llama_index.embeddings.nvidia.base.AsyncOpenAI"
        ) as mock_async_openai_cls,
    ):
        # Set up the sync client mock
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_embeddings.create.return_value = mock_response
        mock_client.embeddings = mock_embeddings
        mock_openai_cls.return_value = mock_client

        # Set up the async client mock properly
        mock_aclient = Mock()
        mock_aembeddings = Mock()
        # Use AsyncMock for the create method to make it awaitable
        mock_aembeddings.create = AsyncMock(return_value=mock_response)
        mock_aclient.embeddings = mock_aembeddings
        mock_async_openai_cls.return_value = mock_aclient

        yield mock_client, mock_aclient


@pytest.mark.parametrize("method_name", ["get_query_embedding", "get_text_embedding"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
def test_single_truncate(method_name: str, truncate: str):
    # Call the method
    getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)("nvidia")


@pytest.mark.parametrize("method_name", ["aget_query_embedding", "aget_text_embedding"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
@pytest.mark.asyncio
async def test_asingle_truncate(method_name: str, truncate: str):
    # Call the method
    await getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)(
        "nvidia"
    )


@pytest.mark.parametrize("method_name", ["get_text_embedding_batch"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
def test_batch_truncate(method_name: str, truncate: str):
    # Call the method
    getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)(
        ["nvidia"]
    )


@pytest.mark.parametrize("method_name", ["aget_text_embedding_batch"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
@pytest.mark.asyncio
async def test_abatch_truncate(method_name: str, truncate: str):
    # Call the method
    await getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)(
        ["nvidia"]
    )
