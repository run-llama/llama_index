import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import requests
from google.genai.errors import APIError
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


EMBEDDING_2_MODEL = "gemini-embedding-2"
LEGACY_MODEL = "gemini-embedding-001"
API_KEY = "fake_key"
EXAMPLE_TEXT = "example"
QUERY_TEXT = "Where is Paris?"
DOCUMENT_TEXT = "Paris is in France."
EMBEDDING_VALUES = [0.1, 0.2, 0.3]
OUTPUT_DIMENSIONALITY = 128
UNKNOWN_TASK_TYPE = "UNKNOWN_TASK"
TASK_INSTRUCTIONS = {
    "RETRIEVAL_QUERY": f"task: search result | query: {EXAMPLE_TEXT}",
    "RETRIEVAL_DOCUMENT": f"title: none | text: {EXAMPLE_TEXT}",
    "QUESTION_ANSWERING": f"task: question answering | query: {EXAMPLE_TEXT}",
    "FACT_VERIFICATION": f"task: fact checking | query: {EXAMPLE_TEXT}",
    "CODE_RETRIEVAL_QUERY": f"task: code retrieval | query: {EXAMPLE_TEXT}",
    "SEMANTIC_SIMILARITY": f"task: sentence similarity | query: {EXAMPLE_TEXT}",
    "CLASSIFICATION": f"task: classification | query: {EXAMPLE_TEXT}",
    "CLUSTERING": f"task: clustering | query: {EXAMPLE_TEXT}",
}


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
    emb = GoogleGenAIEmbedding(model_name=LEGACY_MODEL, api_key=API_KEY)
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


@patch("google.genai.Client")
def test_embedding_2_formats_query_instruction(mock_client_class):
    mock_embed_content = mock_client_class.return_value.models.embed_content
    mock_result = MagicMock()
    mock_result.embeddings = [MagicMock(values=EMBEDDING_VALUES)]
    mock_embed_content.return_value = mock_result

    emb = GoogleGenAIEmbedding(
        model_name=EMBEDDING_2_MODEL,
        api_key=API_KEY,
        embedding_config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": OUTPUT_DIMENSIONALITY,
        },
    )
    emb.get_query_embedding(QUERY_TEXT)

    _, kwargs = mock_embed_content.call_args
    assert kwargs["contents"] == [[f"task: search result | query: {QUERY_TEXT}"]]
    assert kwargs["config"]["output_dimensionality"] == OUTPUT_DIMENSIONALITY
    assert "task_type" not in kwargs["config"]


@patch("google.genai.Client")
def test_default_model_uses_embedding_2_instructions(mock_client_class):
    mock_embed_content = mock_client_class.return_value.models.embed_content
    mock_result = MagicMock()
    mock_result.embeddings = [MagicMock(values=EMBEDDING_VALUES)]
    mock_embed_content.return_value = mock_result

    emb = GoogleGenAIEmbedding(api_key=API_KEY)
    emb.get_query_embedding(QUERY_TEXT)

    _, kwargs = mock_embed_content.call_args
    assert kwargs["model"] == EMBEDDING_2_MODEL
    assert kwargs["contents"] == [[f"task: search result | query: {QUERY_TEXT}"]]


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_embedding_2_formats_document_instruction(mock_client_class):
    mock_aembed_content = AsyncMock()
    mock_result = MagicMock()
    mock_result.embeddings = [MagicMock(values=EMBEDDING_VALUES)]
    mock_aembed_content.return_value = mock_result
    mock_client_class.return_value.aio.models.embed_content = mock_aembed_content

    emb = GoogleGenAIEmbedding(model_name=EMBEDDING_2_MODEL, api_key=API_KEY)
    await emb.aget_text_embedding(DOCUMENT_TEXT)

    _, kwargs = mock_aembed_content.call_args
    assert kwargs["contents"] == [[f"title: none | text: {DOCUMENT_TEXT}"]]
    assert kwargs["config"] is None


@patch("google.genai.Client")
@pytest.mark.parametrize(("task_type", "expected"), TASK_INSTRUCTIONS.items())
def test_embedding_2_formats_all_task_instructions(
    mock_client_class, task_type, expected
):
    emb = GoogleGenAIEmbedding(model_name=EMBEDDING_2_MODEL, api_key=API_KEY)

    texts, _ = emb._prepare_request([EXAMPLE_TEXT], task_type)

    assert texts == [expected]


@patch("google.genai.Client")
def test_embedding_2_uses_configured_task_instruction(mock_client_class):
    mock_embed_content = mock_client_class.return_value.models.embed_content
    mock_result = MagicMock()
    mock_result.embeddings = [MagicMock(values=EMBEDDING_VALUES)]
    mock_embed_content.return_value = mock_result
    emb = GoogleGenAIEmbedding(
        model_name=EMBEDDING_2_MODEL,
        api_key=API_KEY,
        embedding_config={"task_type": "CLASSIFICATION"},
    )

    emb._embed_texts([EXAMPLE_TEXT])

    _, kwargs = mock_embed_content.call_args
    assert kwargs["contents"] == [["task: classification | query: example"]]
    assert kwargs["config"] == {}


@patch("google.genai.Client")
def test_embedding_2_warns_when_task_instruction_is_unknown(mock_client_class):
    emb = GoogleGenAIEmbedding(model_name=EMBEDDING_2_MODEL, api_key=API_KEY)

    with pytest.warns(UserWarning, match=UNKNOWN_TASK_TYPE):
        texts, _ = emb._prepare_request([EXAMPLE_TEXT], UNKNOWN_TASK_TYPE)

    assert texts == ["task: search result | query: example"]


@patch("google.genai.Client")
@pytest.mark.parametrize(
    ("task_type", "text"),
    [
        ("RETRIEVAL_QUERY", "task: search result | query: existing query"),
        ("RETRIEVAL_DOCUMENT", "title: Existing | text: existing document"),
    ],
)
def test_embedding_2_preserves_existing_instruction(mock_client_class, task_type, text):
    emb = GoogleGenAIEmbedding(model_name=EMBEDDING_2_MODEL, api_key=API_KEY)

    texts, _ = emb._prepare_request([text], task_type)

    assert texts == [text]


@patch("google.genai.Client")
def test_legacy_model_preserves_embedding_config(mock_client_class):
    mock_embed_content = mock_client_class.return_value.models.embed_content
    mock_result = MagicMock()
    mock_result.embeddings = [MagicMock(values=EMBEDDING_VALUES)]
    mock_embed_content.return_value = mock_result
    emb = GoogleGenAIEmbedding(
        api_key=API_KEY,
        model_name=LEGACY_MODEL,
        embedding_config={"output_dimensionality": OUTPUT_DIMENSIONALITY},
    )

    emb.get_query_embedding(QUERY_TEXT)
    _, query_kwargs = mock_embed_content.call_args
    assert query_kwargs["config"]["task_type"] == "RETRIEVAL_QUERY"

    emb._embed_texts([DOCUMENT_TEXT])
    _, document_kwargs = mock_embed_content.call_args
    assert document_kwargs["config"].output_dimensionality == OUTPUT_DIMENSIONALITY


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


@patch("google.genai.Client")
def test_client_header_initialization(mock_client_class):
    """Test that the client header is correctly passed to the GoogleGenAIEmbedding."""
    # Setup mock client
    mock_client = mock_client_class.return_value
    mock_client_class.return_value = mock_client

    # Initialize embedding model
    GoogleGenAIEmbedding(api_key="fake_key")

    # Check if http_options were passed to the client constructor
    call_args = mock_client_class.call_args
    _, kwargs = call_args

    http_options = kwargs["http_options"]
    headers = http_options.headers

    assert "x-goog-api-client" in headers
    assert headers["x-goog-api-client"].startswith("llamaindex/")
