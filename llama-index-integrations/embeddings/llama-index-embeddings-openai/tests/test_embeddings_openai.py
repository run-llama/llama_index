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
