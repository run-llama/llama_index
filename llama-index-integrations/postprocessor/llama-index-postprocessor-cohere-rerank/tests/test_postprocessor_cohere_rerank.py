from unittest.mock import MagicMock, patch

import cohere
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.cohere_rerank.base import _create_retry_decorator


def test_class():
    names_of_base_classes = [b.__name__ for b in CohereRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_create_retry_decorator():
    """Test that _create_retry_decorator creates a working decorator."""
    decorator = _create_retry_decorator(max_retries=3)
    assert decorator is not None

    call_count = 0

    @decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise cohere.errors.ServiceUnavailableError(body=None)
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 3


def test_create_retry_decorator_retries_on_internal_server_error():
    """Test that retry decorator retries on InternalServerError."""
    decorator = _create_retry_decorator(max_retries=3)
    call_count = 0

    @decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise cohere.errors.InternalServerError(body=None)
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 2


def test_create_retry_decorator_retries_on_gateway_timeout():
    """Test that retry decorator retries on GatewayTimeoutError."""
    decorator = _create_retry_decorator(max_retries=3)
    call_count = 0

    @decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise cohere.errors.GatewayTimeoutError(body=None)
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 2


def test_rerank_with_retry():
    """Test that _postprocess_nodes uses retry logic."""
    with patch.dict("os.environ", {"COHERE_API_KEY": "test_key"}):
        reranker = CohereRerank(api_key="test_key", max_retries=3)

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.9

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    mock_client = MagicMock()
    mock_client.rerank.return_value = mock_response

    reranker._client = mock_client

    nodes = [NodeWithScore(node=TextNode(text="test document"), score=0.5)]
    query_bundle = QueryBundle(query_str="test query")

    result = reranker._postprocess_nodes(nodes, query_bundle)

    assert len(result) == 1
    assert result[0].score == 0.9
    mock_client.rerank.assert_called_once()


def test_max_retries_parameter():
    """Test that max_retries parameter is properly set."""
    with patch.dict("os.environ", {"COHERE_API_KEY": "test_key"}):
        reranker = CohereRerank(api_key="test_key", max_retries=5)
        assert reranker.max_retries == 5

        reranker_default = CohereRerank(api_key="test_key")
        assert reranker_default.max_retries == 10
