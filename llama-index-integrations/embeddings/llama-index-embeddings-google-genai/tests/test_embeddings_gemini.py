import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import requests
from google.genai.errors import APIError
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


def test_embedding_class():
    emb = GoogleGenAIEmbedding(api_key="...")
    assert isinstance(emb, BaseEmbedding)


# Mock tests that don't require API key
@patch("google.genai.Client")
def test_embed_texts_mock(mock_client_class):
    # Setup mock responses
    mock_client = mock_client_class.return_value
    mock_models = mock_client.models
    mock_embed_content = mock_models.embed_content

    # Create mock embedding result
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]

    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]
    mock_embed_content.return_value = mock_result

    # Test embedding
    emb = GoogleGenAIEmbedding(api_key="fake_key")
    result = emb.get_text_embedding_batch(["test text"])

    # Verify results and calls
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]
    mock_embed_content.assert_called_once()


@patch("google.genai.Client")
def test_task_type_setting_mock(mock_client_class):
    # Setup mock client
    mock_client = mock_client_class.return_value
    mock_models = mock_client.models
    mock_embed_content = mock_models.embed_content

    # Create mock embedding result
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]

    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]
    mock_embed_content.return_value = mock_result

    # Test query embedding (should use RETRIEVAL_QUERY task type)
    emb = GoogleGenAIEmbedding(api_key="fake_key")
    emb.get_query_embedding("test query")

    # Check if task_type was set correctly in the call
    _, kwargs = mock_embed_content.call_args
    assert kwargs.get("config").task_type == "RETRIEVAL_QUERY"

    # Reset mock
    mock_embed_content.reset_mock()

    # Test text embedding (should use RETRIEVAL_DOCUMENT task type)
    emb.get_text_embedding("test text")

    # Check if task_type was set correctly in the call
    _, kwargs = mock_embed_content.call_args
    assert kwargs.get("config").task_type == "RETRIEVAL_DOCUMENT"


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_async_embed_texts_mock(mock_client_class):
    # Setup mock for async client
    mock_client = mock_client_class.return_value
    mock_aio = MagicMock()
    mock_client.aio = mock_aio
    mock_aio_models = mock_aio.models
    mock_aembed_content = mock_aio_models.embed_content

    # Create mock embedding result
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]

    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]

    mock_aembed_content = AsyncMock(return_value=mock_result)
    mock_aio_models.embed_content = mock_aembed_content

    # Test async embedding
    emb = GoogleGenAIEmbedding(api_key="fake_key")
    result = await emb.aget_text_embedding_batch(["test text"])

    # Verify results and calls
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]
    mock_aembed_content.assert_called_once()


# Real API tests (skipped if no API key)
@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None,
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_real_embedding():
    # Initialize with API key from environment
    emb = GoogleGenAIEmbedding()

    # Test query embedding
    query_embedding = emb.get_query_embedding("What is the capital of France?")

    # Simple validation
    assert len(query_embedding) > 0
    assert isinstance(query_embedding, list)
    assert all(isinstance(x, float) for x in query_embedding)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None,
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_real_batch_embedding():
    # Initialize with API key from environment
    emb = GoogleGenAIEmbedding()

    # Test batch embedding
    texts = ["Hello world", "This is a test", "Embeddings are useful"]
    embeddings = emb.get_text_embedding_batch(texts)

    # Validate
    assert len(embeddings) == 3
    assert all(len(emb) > 0 for emb in embeddings)

    # Check that embeddings are different (basic sanity check)
    emb1 = np.array(embeddings[0])
    emb2 = np.array(embeddings[1])
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    assert cos_sim < 0.99  # Different texts should have different embeddings


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None,
    reason="GOOGLE_API_KEY environment variable not set",
)
async def test_real_async_embedding():
    # Initialize with API key from environment
    emb = GoogleGenAIEmbedding()

    # Test async query embedding
    query_embedding = await emb.aget_query_embedding("What is the capital of France?")

    # Simple validation
    assert len(query_embedding) > 0
    assert isinstance(query_embedding, list)
    assert all(isinstance(x, float) for x in query_embedding)


@patch("google.genai.Client")
def test_retry_on_api_error(mock_client_class):
    """Test that the embedding method retries on API rate limit errors."""
    # Setup mock client
    mock_client = mock_client_class.return_value
    mock_models = mock_client.models
    mock_embed_content = mock_models.embed_content

    # Create mock embedding result for successful attempt
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]

    # Make embed_content fail with rate limit error on first call, then succeed
    mock_embed_content.side_effect = [
        APIError(429, response_json={"error": {"message": "Rate limit exceeded"}}),
        mock_result,
    ]

    # Test embedding with retries configured
    emb = GoogleGenAIEmbedding(
        api_key="fake_key",
        retries=2,
        retry_min_seconds=0.1,  # Use small values for faster tests
        retry_max_seconds=0.2,
    )

    # This should fail once, retry, then succeed
    result = emb.get_text_embedding("test text")

    # Verify the result is correct
    assert result == [0.1, 0.2, 0.3]

    # Verify embed_content was called twice (original + 1 retry)
    assert mock_embed_content.call_count == 2


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_async_retry_on_connection_error(mock_client_class):
    """Test that the async embedding method retries on connection errors."""
    # Setup mock for async client
    mock_client = mock_client_class.return_value
    mock_aio = MagicMock()
    mock_client.aio = mock_aio
    mock_aio_models = mock_aio.models

    # Create mock embedding result for successful attempt
    mock_embedding = MagicMock()
    mock_embedding.values = [0.4, 0.5, 0.6]
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]

    # Create two different AsyncMock objects
    fail_mock = AsyncMock(
        side_effect=requests.exceptions.ConnectionError("Connection error")
    )
    success_mock = AsyncMock(return_value=mock_result)

    # Configure the mock to return different mocks on consecutive calls
    mock_aio_models.embed_content = fail_mock

    # Test async embedding with retries
    emb = GoogleGenAIEmbedding(
        api_key="fake_key",
        retries=2,
        retry_min_seconds=0.1,  # Use small values for faster tests
        retry_max_seconds=0.2,
    )

    # Replace the mock after the first call to simulate recovery
    async def side_effect(*args, **kwargs):
        # Replace the mock after first call
        mock_aio_models.embed_content = success_mock
        raise requests.exceptions.ConnectionError("Connection error")

    fail_mock.side_effect = side_effect

    # This should fail once, retry, then succeed
    result = await emb.aget_query_embedding("test query")

    # Verify the result is correct
    assert result == [0.4, 0.5, 0.6]

    # Verify both mocks were called (original + retry)
    fail_mock.assert_called_once()
    success_mock.assert_called_once()


@patch("google.genai.Client")
def test_no_retry_on_auth_error(mock_client_class):
    """Test that authentication errors from invalid API keys are NOT retried."""
    # Setup mock client
    mock_client = mock_client_class.return_value
    mock_models = mock_client.models
    mock_embed_content = mock_models.embed_content

    # Make embed_content fail with authentication error (invalid API key)
    auth_error = APIError(401, response_json={"error": {"message": "Invalid API key"}})
    mock_embed_content.side_effect = auth_error

    # Test embedding with retries configured
    emb = GoogleGenAIEmbedding(
        api_key="invalid_key",
        retries=3,  # Even with multiple retries configured
        retry_min_seconds=0.1,
        retry_max_seconds=0.2,
    )

    # Should raise the APIError without retrying
    with pytest.raises(APIError) as excinfo:
        emb.get_text_embedding("test text")

    # Verify error is the same auth error
    assert excinfo.value == auth_error

    # Verify embed_content was called exactly once (no retries)
    mock_embed_content.assert_called_once()
