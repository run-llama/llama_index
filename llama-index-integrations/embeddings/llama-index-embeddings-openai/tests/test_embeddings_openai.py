import pytest
from unittest.mock import patch

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def test_openai_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenAIEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


@pytest.fixture()
def embedding_instance():
    """Fixture for creating an OpenAIEmbedding instance."""
    return OpenAIEmbedding(
        api_key="test_api_key",
        mode="similarity",
        model="text-embedding-ada-002",
        max_retries=3,
    )


@patch("llama_index.embeddings.openai.base.get_embedding", return_value=[0.1, 0.2, 0.3])
@patch("llama_index.embeddings.openai.base.create_retry_decorator")
def test_get_query_embedding(
    mock_retry_decorator, mock_get_embedding, embedding_instance
):
    """Test _get_query_embedding with retries."""
    # Mock the retry decorator to just return the original function
    mock_retry_decorator.side_effect = lambda **kwargs: (lambda f: f)

    query = "test query"

    result = embedding_instance._get_query_embedding(query)

    # Assert embedding function is called with correct parameters
    mock_get_embedding.assert_called_once_with(
        embedding_instance._get_client(),
        query,
        engine=embedding_instance._query_engine,
        **embedding_instance.additional_kwargs,
    )

    # Assert the result is returned correctly
    assert result == [0.1, 0.2, 0.3]


@patch("llama_index.embeddings.openai.base.get_embedding")
@patch("llama_index.embeddings.openai.base.create_retry_decorator")
def test_get_query_embedding_retry(
    mock_retry_decorator, mock_get_embedding, embedding_instance
):
    """Test retry mechanism in _get_query_embedding."""

    # Mock the retry decorator to simulate retries without delays
    def retry_decorator_simulation(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < embedding_instance.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempts += 1
                    if attempts == embedding_instance.max_retries:
                        raise
            return None

        return wrapper

    mock_retry_decorator.side_effect = lambda **kwargs: retry_decorator_simulation

    # Simulate failures on the first two calls, success on the third
    mock_get_embedding.side_effect = [
        Exception("Temporary error"),  # First call fails
        Exception("Temporary error"),  # Second call fails
        [0.1, 0.2, 0.3],  # Third call succeeds
    ]

    query = "test query"

    # Call the method under test
    result = embedding_instance._get_query_embedding(query)

    # Assert the embedding function is retried and eventually succeeds
    assert mock_get_embedding.call_count == 3
    mock_get_embedding.assert_called_with(
        embedding_instance._get_client(),
        query,
        engine=embedding_instance._query_engine,
        **embedding_instance.additional_kwargs,
    )

    # Assert the result matches the mock output
    assert result == [0.1, 0.2, 0.3]


@patch("llama_index.embeddings.openai.base.get_embedding")
@patch("llama_index.embeddings.openai.base.create_retry_decorator")
def test_get_query_embedding_retry_exhausted(
    mock_retry_decorator, mock_get_embedding, embedding_instance
):
    """Test retry mechanism exhausts all retries in _get_query_embedding."""

    # Mock the retry decorator to simulate retries without delays
    def retry_decorator_simulation(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < embedding_instance.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempts += 1
                    if attempts == embedding_instance.max_retries:
                        raise

            return None

        return wrapper

    mock_retry_decorator.side_effect = lambda **kwargs: retry_decorator_simulation

    # Simulate persistent failures
    mock_get_embedding.side_effect = Exception("Persistent error")

    query = "test query"

    # Call the method under test and expect an exception
    with pytest.raises(Exception, match="Persistent error"):
        embedding_instance._get_query_embedding(query)

    # Assert the embedding function was retried the maximum number of times
    assert mock_get_embedding.call_count == embedding_instance.max_retries
    mock_get_embedding.assert_called_with(
        embedding_instance._get_client(),
        query,
        engine=embedding_instance._query_engine,
        **embedding_instance.additional_kwargs,
    )


@patch("llama_index.embeddings.openai.base.get_embedding", return_value=[0.1, 0.2, 0.3])
@patch("llama_index.embeddings.openai.base.create_retry_decorator")
def test_get_text_embedding(
    mock_retry_decorator, mock_get_embedding, embedding_instance
):
    """Test _get_text_embedding with retries."""
    # Mock the retry decorator to accept any arguments and return the original function
    mock_retry_decorator.side_effect = lambda **kwargs: (lambda f: f)

    text = "test text"

    # Call the method under test
    result = embedding_instance._get_text_embedding(text)

    # Assert the get_embedding function is called with correct parameters
    mock_get_embedding.assert_called_once_with(
        embedding_instance._get_client(),
        text,
        engine=embedding_instance._text_engine,
        **embedding_instance.additional_kwargs,
    )

    # Assert the result is returned correctly
    assert result == [0.1, 0.2, 0.3]


@patch(
    "llama_index.embeddings.openai.base.get_embeddings",
    return_value=[[0.1, 0.2], [0.3, 0.4]],
)
@patch("llama_index.embeddings.openai.base.create_retry_decorator")
def test_get_text_embeddings(
    mock_retry_decorator, mock_get_embeddings, embedding_instance
):
    """Test _get_text_embeddings with retries."""
    # Mock the retry decorator to accept any arguments and return the original function
    mock_retry_decorator.side_effect = lambda **kwargs: (lambda f: f)

    texts = ["test text 1", "test text 2"]

    result = embedding_instance._get_text_embeddings(texts)

    # Assert embedding function is called with correct parameters
    mock_get_embeddings.assert_called_once_with(
        embedding_instance._get_client(),
        texts,
        engine=embedding_instance._text_engine,
        **embedding_instance.additional_kwargs,
    )

    # Assert the result is returned correctly
    assert result == [[0.1, 0.2], [0.3, 0.4]]
