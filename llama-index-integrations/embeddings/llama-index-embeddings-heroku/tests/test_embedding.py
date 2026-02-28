"""Tests for HerokuEmbedding."""

import pytest
from pytest_httpx import HTTPXMock

from llama_index.embeddings.heroku import HerokuEmbedding


class TestHerokuEmbedding:
    """Test suite for HerokuEmbedding class."""

    @pytest.fixture
    def embed_model(self) -> HerokuEmbedding:
        """Create an embedding model for testing."""
        return HerokuEmbedding(
            api_key="test-api-key",
            model="cohere-embed-multilingual",
        )

    @pytest.fixture
    def mock_embedding_response(self) -> dict:
        """Create a mock embedding response."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1024,
                }
            ],
            "model": "cohere-embed-multilingual",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    @pytest.fixture
    def mock_batch_embedding_response(self) -> dict:
        """Create a mock batch embedding response."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1024,
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [0.2] * 1024,
                },
            ],
            "model": "cohere-embed-multilingual",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

    def test_initialization(self, embed_model: HerokuEmbedding) -> None:
        """Test embedding model initialization."""
        assert embed_model.api_key == "test-api-key"
        assert embed_model.model == "cohere-embed-multilingual"
        assert embed_model.base_url == "https://us.inference.heroku.com"
        assert embed_model.embed_batch_size == 96

    def test_class_name(self, embed_model: HerokuEmbedding) -> None:
        """Test class name method."""
        assert embed_model.class_name() == "HerokuEmbedding"

    def test_get_text_embedding(
        self,
        httpx_mock: HTTPXMock,
        embed_model: HerokuEmbedding,
        mock_embedding_response: dict,
    ) -> None:
        """Test getting a single text embedding."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )

        embedding = embed_model._get_text_embedding("Hello, world!")

        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

    def test_get_text_embeddings(
        self,
        httpx_mock: HTTPXMock,
        embed_model: HerokuEmbedding,
        mock_batch_embedding_response: dict,
    ) -> None:
        """Test getting multiple text embeddings."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/embeddings",
            method="POST",
            json=mock_batch_embedding_response,
        )

        embeddings = embed_model._get_text_embeddings(["Hello", "World"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024
        assert len(embeddings[1]) == 1024

    def test_get_query_embedding(
        self,
        httpx_mock: HTTPXMock,
        embed_model: HerokuEmbedding,
        mock_embedding_response: dict,
    ) -> None:
        """Test getting a query embedding."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )

        embedding = embed_model._get_query_embedding("What is the answer?")

        assert len(embedding) == 1024

    @pytest.mark.asyncio
    async def test_aget_text_embedding(
        self,
        httpx_mock: HTTPXMock,
        embed_model: HerokuEmbedding,
        mock_embedding_response: dict,
    ) -> None:
        """Test async getting a single text embedding."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )

        embedding = await embed_model._aget_text_embedding("Hello, world!")

        assert len(embedding) == 1024

    @pytest.mark.asyncio
    async def test_aget_text_embeddings(
        self,
        httpx_mock: HTTPXMock,
        embed_model: HerokuEmbedding,
        mock_batch_embedding_response: dict,
    ) -> None:
        """Test async getting multiple text embeddings."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/embeddings",
            method="POST",
            json=mock_batch_embedding_response,
        )

        embeddings = await embed_model._aget_text_embeddings(["Hello", "World"])

        assert len(embeddings) == 2

    def test_request_headers(self, embed_model: HerokuEmbedding) -> None:
        """Test that correct headers are generated."""
        headers = embed_model._get_headers()

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_request_body(self, embed_model: HerokuEmbedding) -> None:
        """Test that correct request body is generated."""
        body = embed_model._get_request_body(["test text"])

        assert body["model"] == "cohere-embed-multilingual"
        assert body["input"] == ["test text"]

    def test_request_body_with_input_type(self) -> None:
        """Test request body with input_type set."""
        embed_model = HerokuEmbedding(
            api_key="test-key",
            input_type="search_document",
        )
        body = embed_model._get_request_body(["test"])

        assert body["input_type"] == "search_document"

    def test_custom_base_url(self) -> None:
        """Test custom base URL configuration."""
        embed_model = HerokuEmbedding(
            api_key="test-key",
            base_url="https://custom.heroku.com",
        )

        assert embed_model.base_url == "https://custom.heroku.com"

    def test_embeddings_sorted_by_index(
        self,
        httpx_mock: HTTPXMock,
        embed_model: HerokuEmbedding,
    ) -> None:
        """Test that embeddings are sorted by index."""
        # Response with out-of-order indices
        response = {
            "data": [
                {"index": 1, "embedding": [0.2] * 1024},
                {"index": 0, "embedding": [0.1] * 1024},
            ]
        }
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/embeddings",
            method="POST",
            json=response,
        )

        embeddings = embed_model._get_text_embeddings(["first", "second"])

        # First embedding should have 0.1 values, second should have 0.2
        assert embeddings[0][0] == 0.1
        assert embeddings[1][0] == 0.2
