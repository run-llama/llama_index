"""Test Isaacus embeddings."""

import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.embeddings.isaacus.base import IsaacusEmbedding

STUB_MODEL = "kanon-2-embedder"
STUB_API_KEY = "test-api-key"
STUB_BASE_URL = "https://api.isaacus.com/v1"


@pytest.fixture(name="isaacus_embedding")
def fixture_isaacus_embedding() -> IsaacusEmbedding:
    """Create an IsaacusEmbedding instance for testing."""
    return IsaacusEmbedding(
        model=STUB_MODEL,
        api_key=STUB_API_KEY,
        base_url=STUB_BASE_URL,
    )


@pytest.fixture(name="mock_embedding_object")
def fixture_mock_embedding_object() -> MagicMock:
    """Create a mock embedding object."""
    mock_obj = MagicMock()
    mock_obj.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_obj.index = 0
    return mock_obj


@pytest.fixture(name="mock_response")
def fixture_mock_response(mock_embedding_object: MagicMock) -> MagicMock:
    """Create a mock response for testing."""
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding_object]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 5
    return mock_response


class TestIsaacusEmbedding:
    """Test IsaacusEmbedding class."""

    def test_class_name(self, isaacus_embedding: IsaacusEmbedding) -> None:
        """Test class name."""
        assert IsaacusEmbedding.class_name() == "IsaacusEmbedding"
        assert isaacus_embedding.class_name() == "IsaacusEmbedding"

    def test_init_with_parameters(self) -> None:
        """Test initialization with parameters."""
        embedding = IsaacusEmbedding(
            model=STUB_MODEL,
            api_key=STUB_API_KEY,
            base_url=STUB_BASE_URL,
            dimensions=1024,
            task="retrieval/document",
            overflow_strategy="drop_end",
            timeout=30.0,
        )
        assert embedding.model == STUB_MODEL
        assert embedding.api_key == STUB_API_KEY
        assert embedding.base_url == STUB_BASE_URL
        assert embedding.dimensions == 1024
        assert embedding.task == "retrieval/document"
        assert embedding.overflow_strategy == "drop_end"
        assert embedding.timeout == 30.0

    def test_init_with_environment_variables(self) -> None:
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "ISAACUS_API_KEY": STUB_API_KEY,
                "ISAACUS_BASE_URL": STUB_BASE_URL,
            },
        ):
            embedding = IsaacusEmbedding()
            assert embedding.model == STUB_MODEL
            assert embedding.api_key == STUB_API_KEY
            assert embedding.base_url == STUB_BASE_URL

    def test_init_missing_api_key(self) -> None:
        """Test initialization with missing API key."""
        with pytest.raises(ValueError, match="API key is required"):
            IsaacusEmbedding(
                base_url=STUB_BASE_URL,
            )

    def test_get_text_embedding_success(
        self, isaacus_embedding: IsaacusEmbedding, mock_response: MagicMock
    ) -> None:
        """Test successful text embedding."""
        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ):
            embedding = isaacus_embedding.get_text_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_get_text_embedding_with_task(
        self, isaacus_embedding: IsaacusEmbedding, mock_response: MagicMock
    ) -> None:
        """Test text embedding with task parameter."""
        isaacus_embedding.task = "retrieval/document"
        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ) as mock_create:
            embedding = isaacus_embedding.get_text_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            # Verify task was passed to API
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["task"] == "retrieval/document"

    def test_get_query_embedding_uses_retrieval_query_task(
        self, isaacus_embedding: IsaacusEmbedding, mock_response: MagicMock
    ) -> None:
        """Test that get_query_embedding uses retrieval/query task."""
        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ) as mock_create:
            embedding = isaacus_embedding.get_query_embedding("test query")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            # Verify task was set to retrieval/query
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["task"] == "retrieval/query"

    def test_get_text_embedding_error(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test text embedding with error."""
        with patch.object(
            isaacus_embedding._client.embeddings,
            "create",
            side_effect=Exception("API error"),
        ):
            with pytest.raises(ValueError, match="Unable to embed text"):
                isaacus_embedding.get_text_embedding("test text")

    def test_get_text_embedding_no_embeddings_returned(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test text embedding when no embeddings are returned."""
        mock_response = MagicMock()
        mock_response.embeddings = []

        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ):
            with pytest.raises(ValueError, match="No embeddings returned from API"):
                isaacus_embedding.get_text_embedding("test text")

    def test_get_text_embeddings_batch(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test batch text embeddings."""
        # Create mock response with multiple embeddings
        mock_emb1 = MagicMock()
        mock_emb1.embedding = [0.1, 0.2, 0.3]
        mock_emb1.index = 0

        mock_emb2 = MagicMock()
        mock_emb2.embedding = [0.4, 0.5, 0.6]
        mock_emb2.index = 1

        mock_emb3 = MagicMock()
        mock_emb3.embedding = [0.7, 0.8, 0.9]
        mock_emb3.index = 2

        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb1, mock_emb2, mock_emb3]

        texts = ["text1", "text2", "text3"]
        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ):
            embeddings = isaacus_embedding.get_text_embedding_batch(texts)
            assert len(embeddings) == 3
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            assert embeddings[2] == [0.7, 0.8, 0.9]

    def test_get_text_embeddings_maintains_order(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test that batch embeddings maintain correct order."""
        # Create mock response with embeddings out of order
        mock_emb1 = MagicMock()
        mock_emb1.embedding = [0.1, 0.2, 0.3]
        mock_emb1.index = 0

        mock_emb2 = MagicMock()
        mock_emb2.embedding = [0.4, 0.5, 0.6]
        mock_emb2.index = 1

        mock_response = MagicMock()
        # Return embeddings out of order
        mock_response.embeddings = [mock_emb2, mock_emb1]

        texts = ["text1", "text2"]
        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ):
            embeddings = isaacus_embedding.get_text_embedding_batch(texts)
            # Should be sorted by index
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_aget_text_embedding_success(
        self, isaacus_embedding: IsaacusEmbedding, mock_response: MagicMock
    ) -> None:
        """Test successful async text embedding."""
        with patch.object(
            isaacus_embedding._aclient.embeddings, "create", return_value=mock_response
        ):
            embedding = await isaacus_embedding.aget_text_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_aget_query_embedding_uses_retrieval_query_task(
        self, isaacus_embedding: IsaacusEmbedding, mock_response: MagicMock
    ) -> None:
        """Test that aget_query_embedding uses retrieval/query task."""
        with patch.object(
            isaacus_embedding._aclient.embeddings, "create", return_value=mock_response
        ) as mock_create:
            embedding = await isaacus_embedding.aget_query_embedding("test query")
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            # Verify task was set to retrieval/query
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["task"] == "retrieval/query"

    @pytest.mark.asyncio
    async def test_aget_text_embedding_error(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test async text embedding with error."""
        with patch.object(
            isaacus_embedding._aclient.embeddings,
            "create",
            side_effect=Exception("API error"),
        ):
            with pytest.raises(ValueError, match="Unable to embed text"):
                await isaacus_embedding.aget_text_embedding("test text")

    @pytest.mark.asyncio
    async def test_aget_text_embeddings_batch(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test async batch text embeddings."""
        # Create mock response with multiple embeddings
        mock_emb1 = MagicMock()
        mock_emb1.embedding = [0.1, 0.2, 0.3]
        mock_emb1.index = 0

        mock_emb2 = MagicMock()
        mock_emb2.embedding = [0.4, 0.5, 0.6]
        mock_emb2.index = 1

        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb1, mock_emb2]

        texts = ["text1", "text2"]
        with patch.object(
            isaacus_embedding._aclient.embeddings, "create", return_value=mock_response
        ):
            embeddings = await isaacus_embedding.aget_text_embedding_batch(texts)
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_prepare_request_params_basic(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test _prepare_request_params with basic parameters."""
        params = isaacus_embedding._prepare_request_params("test text")
        assert params["model"] == STUB_MODEL
        assert params["texts"] == "test text"
        assert "task" not in params  # No task set by default

    def test_prepare_request_params_with_all_options(self) -> None:
        """Test _prepare_request_params with all options set."""
        embedding = IsaacusEmbedding(
            model=STUB_MODEL,
            api_key=STUB_API_KEY,
            base_url=STUB_BASE_URL,
            dimensions=1024,
            task="retrieval/document",
            overflow_strategy="drop_end",
        )
        params = embedding._prepare_request_params("test text")
        assert params["model"] == STUB_MODEL
        assert params["texts"] == "test text"
        assert params["task"] == "retrieval/document"
        assert params["dimensions"] == 1024
        assert params["overflow_strategy"] == "drop_end"

    def test_prepare_request_params_task_override(
        self, isaacus_embedding: IsaacusEmbedding
    ) -> None:
        """Test _prepare_request_params with task override."""
        isaacus_embedding.task = "retrieval/document"
        params = isaacus_embedding._prepare_request_params(
            "test text", task_override="retrieval/query"
        )
        # Override should take precedence
        assert params["task"] == "retrieval/query"

    def test_embedding_dimensions(self, isaacus_embedding: IsaacusEmbedding) -> None:
        """Test that embeddings have the expected dimensions."""
        mock_emb = MagicMock()
        mock_emb.embedding = [0.1] * 1792  # Default Kanon 2 dimension
        mock_emb.index = 0

        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb]

        with patch.object(
            isaacus_embedding._client.embeddings, "create", return_value=mock_response
        ):
            embedding = isaacus_embedding.get_text_embedding("test text")
            assert len(embedding) == 1792
            assert all(isinstance(x, float) for x in embedding)
