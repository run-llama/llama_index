"""Test Heroku embeddings."""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from llama_index.embeddings.heroku.base import HerokuEmbedding

STUB_MODEL = "cohere-embed-multilingual-v3"
STUB_API_KEY = "test-api-key"
STUB_EMBEDDING_URL = "https://test-inference.heroku.com"


@pytest.fixture(name="heroku_embedding")
def fixture_heroku_embedding() -> HerokuEmbedding:
    """Create a HerokuEmbedding instance for testing."""
    return HerokuEmbedding(
        model=STUB_MODEL,
        api_key=STUB_API_KEY,
        base_url=STUB_EMBEDDING_URL,
    )


@pytest.fixture(name="mock_response")
def fixture_mock_response() -> MagicMock:
    """Create a mock response for testing."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "index": 0,
                "object": "embedding",
            }
        ],
        "model": STUB_MODEL,
        "object": "list",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


class TestHerokuEmbedding:
    """Test HerokuEmbedding class."""

    def test_class_name(self, heroku_embedding: HerokuEmbedding) -> None:
        """Test class name."""
        assert HerokuEmbedding.class_name() == "HerokuEmbedding"
        assert heroku_embedding.class_name() == "HerokuEmbedding"

    def test_init_with_parameters(self) -> None:
        """Test initialization with parameters."""
        embedding = HerokuEmbedding(
            model=STUB_MODEL,
            api_key=STUB_API_KEY,
            base_url=STUB_EMBEDDING_URL,
            timeout=30.0,
        )
        assert embedding.model == STUB_MODEL
        assert embedding.api_key == STUB_API_KEY
        assert embedding.base_url == STUB_EMBEDDING_URL
        assert embedding.timeout == 30.0

    def test_init_with_environment_variables(self) -> None:
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_KEY": STUB_API_KEY,
                "EMBEDDING_URL": STUB_EMBEDDING_URL,
                "EMBEDDING_MODEL_ID": STUB_MODEL,
            },
        ):
            embedding = HerokuEmbedding()
            assert embedding.model == STUB_MODEL
            assert embedding.api_key == STUB_API_KEY
            assert embedding.base_url == STUB_EMBEDDING_URL

    def test_init_missing_api_key(self) -> None:
        """Test initialization with missing API key."""
        with pytest.raises(ValueError, match="API key is required"):
            HerokuEmbedding(
                model=STUB_MODEL,
                base_url=STUB_EMBEDDING_URL,
            )

    def test_init_missing_base_url(self) -> None:
        """Test initialization with missing embedding URL."""
        with pytest.raises(ValueError, match="Embedding URL is required"):
            HerokuEmbedding(
                model=STUB_MODEL,
                api_key=STUB_API_KEY,
            )

    def test_init_missing_model(self) -> None:
        """Test initialization with missing model."""
        with pytest.raises(ValueError, match="Model is required"):
            HerokuEmbedding(
                api_key=STUB_API_KEY,
                base_url=STUB_EMBEDDING_URL,
            )

    def test_get_text_embedding_success(
        self, heroku_embedding: HerokuEmbedding, mock_response: MagicMock
    ) -> None:
        """Test successful text embedding."""
        with patch.object(heroku_embedding._client, "post", return_value=mock_response):
            embedding = heroku_embedding.get_text_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_get_text_embedding_http_error(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test text embedding with HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )

        with patch.object(heroku_embedding._client, "post", return_value=mock_response):
            with pytest.raises(ValueError, match="Unable to embed text"):
                heroku_embedding.get_text_embedding("test text")

    def test_get_text_embedding_exception(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test text embedding with general exception."""
        with patch.object(
            heroku_embedding._client, "post", side_effect=Exception("Network error")
        ):
            with pytest.raises(ValueError, match="Unable to embed text"):
                heroku_embedding.get_text_embedding("test text")

    def test_get_query_embedding(self, heroku_embedding: HerokuEmbedding) -> None:
        """Test query embedding."""
        with patch.object(
            heroku_embedding, "_get_text_embedding", return_value=[0.1, 0.2, 0.3]
        ):
            embedding = heroku_embedding.get_query_embedding("test query")
            assert embedding == [0.1, 0.2, 0.3]

    def test_get_text_embeddings(self, heroku_embedding: HerokuEmbedding) -> None:
        """Test batch text embeddings."""
        texts = ["text1", "text2", "text3"]
        with patch.object(
            heroku_embedding, "_get_text_embedding", return_value=[0.1, 0.2, 0.3]
        ):
            embeddings = heroku_embedding.get_text_embedding_batch(texts)
            assert len(embeddings) == 3
            assert all(embedding == [0.1, 0.2, 0.3] for embedding in embeddings)

    @pytest.mark.asyncio
    async def test_aget_text_embedding_success(
        self, heroku_embedding: HerokuEmbedding, mock_response: MagicMock
    ) -> None:
        """Test successful async text embedding."""
        with patch.object(
            heroku_embedding._aclient, "post", return_value=mock_response
        ):
            embedding = await heroku_embedding.aget_text_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_aget_text_embedding_http_error(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test async text embedding with HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )

        with patch.object(
            heroku_embedding._aclient, "post", return_value=mock_response
        ):
            with pytest.raises(ValueError, match="Unable to embed text"):
                await heroku_embedding.aget_text_embedding("test text")

    @pytest.mark.asyncio
    async def test_aget_text_embedding_exception(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test async text embedding with general exception."""
        with patch.object(
            heroku_embedding._aclient, "post", side_effect=Exception("Network error")
        ):
            with pytest.raises(ValueError, match="Unable to embed text"):
                await heroku_embedding.aget_text_embedding("test text")

    @pytest.mark.asyncio
    async def test_aget_query_embedding(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test async query embedding."""
        with patch.object(
            heroku_embedding, "_aget_text_embedding", return_value=[0.1, 0.2, 0.3]
        ):
            embedding = await heroku_embedding.aget_query_embedding("test query")
            assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aget_text_embeddings(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test async batch text embeddings."""
        texts = ["text1", "text2", "text3"]
        with patch.object(
            heroku_embedding, "_aget_text_embedding", return_value=[0.1, 0.2, 0.3]
        ):
            embeddings = await heroku_embedding.aget_text_embedding_batch(texts)
            assert len(embeddings) == 3
            assert all(embedding == [0.1, 0.2, 0.3] for embedding in embeddings)

    def test_cleanup_sync_client(self) -> None:
        """Test cleanup of sync client."""
        embedding = HerokuEmbedding(
            model=STUB_MODEL,
            api_key=STUB_API_KEY,
            base_url=STUB_EMBEDDING_URL,
        )
        with patch.object(embedding._client, "close") as mock_close:
            del embedding
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_async_client(self) -> None:
        """Test cleanup of async client."""
        embedding = HerokuEmbedding(
            model=STUB_MODEL,
            api_key=STUB_API_KEY,
            base_url=STUB_EMBEDDING_URL,
        )
        with patch.object(embedding._aclient, "aclose") as mock_aclose:
            await embedding.aclose()
            mock_aclose.assert_called_once()

    def test_embedding_dimensions(self, heroku_embedding: HerokuEmbedding) -> None:
        """Test that embeddings have the expected dimensions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "embedding": [0.1] * 768,  # Common embedding dimension
                    "index": 0,
                    "object": "embedding",
                }
            ],
            "model": STUB_MODEL,
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(heroku_embedding._client, "post", return_value=mock_response):
            embedding = heroku_embedding.get_text_embedding("test text")
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    def test_batch_embedding_consistency(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test that batch embeddings are consistent."""
        texts = ["text1", "text2"]
        mock_embedding = [0.1, 0.2, 0.3]

        with patch.object(
            heroku_embedding, "_get_text_embedding", return_value=mock_embedding
        ):
            embeddings = heroku_embedding.get_text_embedding_batch(texts)
            assert len(embeddings) == 2
            assert embeddings[0] == embeddings[1] == mock_embedding

    @pytest.mark.asyncio
    async def test_async_batch_embedding_consistency(
        self, heroku_embedding: HerokuEmbedding
    ) -> None:
        """Test that async batch embeddings are consistent."""
        texts = ["text1", "text2"]
        mock_embedding = [0.1, 0.2, 0.3]

        with patch.object(
            heroku_embedding, "_aget_text_embedding", return_value=mock_embedding
        ):
            embeddings = await heroku_embedding.aget_text_embedding_batch(texts)
            assert len(embeddings) == 2
            assert embeddings[0] == embeddings[1] == mock_embedding
