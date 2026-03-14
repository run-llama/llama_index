"""Tests for HerokuRerank."""

import pytest
from pytest_httpx import HTTPXMock

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.heroku_rerank import HerokuRerank


class TestHerokuRerank:
    """Test suite for HerokuRerank class."""

    @pytest.fixture
    def reranker(self) -> HerokuRerank:
        """Create a reranker for testing."""
        return HerokuRerank(
            api_key="test-api-key",
            model="cohere-rerank-3-5",
            top_n=3,
        )

    @pytest.fixture
    def sample_nodes(self) -> list[NodeWithScore]:
        """Create sample nodes for testing."""
        return [
            NodeWithScore(
                node=TextNode(text="First document about Python programming."),
                score=0.8,
            ),
            NodeWithScore(
                node=TextNode(text="Second document about JavaScript."),
                score=0.7,
            ),
            NodeWithScore(
                node=TextNode(text="Third document about Python data science."),
                score=0.6,
            ),
            NodeWithScore(
                node=TextNode(text="Fourth document about web development."),
                score=0.5,
            ),
        ]

    @pytest.fixture
    def mock_rerank_response(self) -> dict:
        """Create a mock rerank response."""
        return {
            "id": "rerank-123",
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 3, "relevance_score": 0.45},
            ],
            "meta": {"api_version": {"version": "1"}},
        }

    def test_initialization(self, reranker: HerokuRerank) -> None:
        """Test reranker initialization."""
        assert reranker.api_key == "test-api-key"
        assert reranker.model == "cohere-rerank-3-5"
        assert reranker.top_n == 3
        assert reranker.base_url == "https://us.inference.heroku.com"

    def test_class_name(self, reranker: HerokuRerank) -> None:
        """Test class name method."""
        assert reranker.class_name() == "HerokuRerank"

    def test_postprocess_nodes(
        self,
        httpx_mock: HTTPXMock,
        reranker: HerokuRerank,
        sample_nodes: list[NodeWithScore],
        mock_rerank_response: dict,
    ) -> None:
        """Test postprocessing nodes with reranking."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/rerank",
            method="POST",
            json=mock_rerank_response,
        )

        query_bundle = QueryBundle(query_str="Python programming tutorial")
        reranked = reranker._postprocess_nodes(sample_nodes, query_bundle)

        assert len(reranked) == 3
        # Check that scores are from the rerank response
        assert reranked[0].score == 0.95
        assert reranked[1].score == 0.85
        assert reranked[2].score == 0.45
        # Check that nodes are reordered correctly
        assert "Python data science" in reranked[0].node.get_content()
        assert "Python programming" in reranked[1].node.get_content()
        assert "web development" in reranked[2].node.get_content()

    def test_postprocess_nodes_empty_list(self, reranker: HerokuRerank) -> None:
        """Test postprocessing with empty node list."""
        query_bundle = QueryBundle(query_str="test query")
        result = reranker._postprocess_nodes([], query_bundle)

        assert result == []

    def test_postprocess_nodes_no_query(
        self, reranker: HerokuRerank, sample_nodes: list[NodeWithScore]
    ) -> None:
        """Test postprocessing without query bundle."""
        result = reranker._postprocess_nodes(sample_nodes, None)

        assert result == sample_nodes

    @pytest.mark.asyncio
    async def test_apostprocess_nodes(
        self,
        httpx_mock: HTTPXMock,
        reranker: HerokuRerank,
        sample_nodes: list[NodeWithScore],
        mock_rerank_response: dict,
    ) -> None:
        """Test async postprocessing nodes."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/rerank",
            method="POST",
            json=mock_rerank_response,
        )

        query_bundle = QueryBundle(query_str="Python programming tutorial")
        reranked = await reranker._apostprocess_nodes(sample_nodes, query_bundle)

        assert len(reranked) == 3
        assert reranked[0].score == 0.95

    def test_request_headers(self, reranker: HerokuRerank) -> None:
        """Test that correct headers are generated."""
        headers = reranker._get_headers()

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_top_n_limits_results(
        self,
        httpx_mock: HTTPXMock,
        sample_nodes: list[NodeWithScore],
    ) -> None:
        """Test that top_n parameter limits results."""
        reranker = HerokuRerank(api_key="test-key", top_n=2)
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ]
        }
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/rerank",
            method="POST",
            json=mock_response,
        )

        query_bundle = QueryBundle(query_str="test")
        reranked = reranker._postprocess_nodes(sample_nodes, query_bundle)

        assert len(reranked) == 2

    def test_custom_base_url(self) -> None:
        """Test custom base URL configuration."""
        reranker = HerokuRerank(
            api_key="test-key",
            base_url="https://custom.heroku.com",
        )

        assert reranker.base_url == "https://custom.heroku.com"

    def test_fewer_nodes_than_top_n(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test reranking when fewer nodes than top_n."""
        reranker = HerokuRerank(api_key="test-key", top_n=10)
        nodes = [
            NodeWithScore(node=TextNode(text="Only one node."), score=0.5),
        ]
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
            ]
        }
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/rerank",
            method="POST",
            json=mock_response,
        )

        query_bundle = QueryBundle(query_str="test")
        reranked = reranker._postprocess_nodes(nodes, query_bundle)

        assert len(reranked) == 1
        assert reranked[0].score == 0.9
