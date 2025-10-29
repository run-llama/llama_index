from unittest.mock import Mock

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.embeddings.voyageai.base import CONTEXT_MODELS


def test_embedding_class():
    emb = VoyageEmbedding(model_name="", voyage_api_key="NOT_A_VALID_KEY")
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 1000
    assert emb.model_name == ""


def test_embedding_class_voyage_2():
    emb = VoyageEmbedding(
        model_name="voyage-2", voyage_api_key="NOT_A_VALID_KEY", truncation=True
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 1000
    assert emb.model_name == "voyage-2"
    assert emb.truncation
    assert emb.output_dimension is None
    assert emb.output_dtype is None


def test_embedding_class_voyage_2_with_batch_size():
    emb = VoyageEmbedding(
        model_name="voyage-2", voyage_api_key="NOT_A_VALID_KEY", embed_batch_size=49
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 49
    assert emb.model_name == "voyage-2"
    assert emb.truncation is None
    assert emb.output_dimension is None
    assert emb.output_dtype is None


def test_embedding_class_voyage_3_large_with_output_dimension():
    emb = VoyageEmbedding(
        model_name="voyage-3-large",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dimension=512,
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 1000
    assert emb.model_name == "voyage-3-large"
    assert emb.truncation is None
    assert emb.output_dimension == 512
    assert emb.output_dtype is None


def test_embedding_class_voyage_3_large_with_output_dtype():
    emb = VoyageEmbedding(
        model_name="voyage-3-large",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dtype="float",
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 1000
    assert emb.model_name == "voyage-3-large"
    assert emb.truncation is None
    assert emb.output_dimension is None
    assert emb.output_dtype == "float"


def test_voyageai_embedding_class():
    names_of_base_classes = [b.__name__ for b in VoyageEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_embedding_class_context_model():
    emb = VoyageEmbedding(
        model_name="voyage-context-3",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dimension=1024,
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 1000
    assert emb.model_name == "voyage-context-3"
    assert emb.output_dimension == 1024
    assert emb.output_dtype is None


def test_embedding_class_context_model_with_params():
    emb = VoyageEmbedding(
        model_name="voyage-context-3",
        voyage_api_key="NOT_A_VALID_KEY",
        truncation=True,
        output_dtype="float",
        output_dimension=512,
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 1000
    assert emb.model_name == "voyage-context-3"
    assert emb.output_dimension == 512
    assert emb.output_dtype == "float"


def test_context_model_detection():
    context_emb = VoyageEmbedding(
        model_name="voyage-context-3", voyage_api_key="NOT_A_VALID_KEY"
    )
    regular_emb = VoyageEmbedding(
        model_name="voyage-3", voyage_api_key="NOT_A_VALID_KEY"
    )

    assert context_emb.model_name in CONTEXT_MODELS
    assert regular_emb.model_name not in CONTEXT_MODELS


# Unit tests for _build_batches method


def test_build_batches_basic():
    """Test basic batch building."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=2,
    )

    # Mock tokenize to return predictable token counts
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4"]
    batches = list(emb._build_batches(texts))

    # Should create 2 batches of 2 texts each
    assert len(batches) == 2
    assert batches[0] == (["text1", "text2"], 2)
    assert batches[1] == (["text3", "text4"], 2)


def test_build_batches_token_limit():
    """Test batch building respects token limits."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=10,
    )

    # Mock tokenize to return large token counts that exceed limits
    # Each text has 200k tokens, so with 320k limit, only 1 text per batch
    mock_tokenize = Mock(return_value=[[1] * 200000])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3"]
    batches = list(emb._build_batches(texts))

    # Token limit for voyage-2 is 320,000
    # Each text has 200k tokens, so should create 3 separate batches
    assert len(batches) == 3
    assert batches[0] == (["text1"], 1)
    assert batches[1] == (["text2"], 1)
    assert batches[2] == (["text3"], 1)


def test_build_batches_single_text():
    """Test batch building with single text."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=10,
    )

    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["single text"]
    batches = list(emb._build_batches(texts))

    assert len(batches) == 1
    assert batches[0] == (["single text"], 1)


def test_build_batches_empty_list():
    """Test batch building with empty list."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
    )

    texts = []
    batches = list(emb._build_batches(texts))

    assert len(batches) == 0


def test_build_batches_respects_max_batch_size():
    """Test that batches never exceed MAX_BATCH_SIZE (1000)."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=1000,  # Default, but should cap at 1000
    )

    # Mock tokenize to return small token counts
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    # Create 1500 texts
    texts = [f"text{i}" for i in range(1500)]
    batches = list(emb._build_batches(texts))

    # Should have at least 2 batches (1500 texts / 1000 max = 1.5)
    assert len(batches) >= 2

    # No batch should exceed 1000 items
    for batch_texts, batch_size in batches:
        assert len(batch_texts) <= 1000
        assert batch_size <= 1000


def test_build_batches_context_model_token_limit():
    """Test batch building with context model's smaller token limit."""
    emb = VoyageEmbedding(
        model_name="voyage-context-3",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=100,
    )

    # Mock tokenize to return 20k tokens per text
    # Context-3 has 32k token limit, so should fit 1 text per batch
    mock_tokenize = Mock(return_value=[[1] * 20000])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3"]
    batches = list(emb._build_batches(texts))

    # With 20k tokens each and 32k limit, should create 3 batches
    assert len(batches) == 3
    assert batches[0] == (["text1"], 1)
    assert batches[1] == (["text2"], 1)
    assert batches[2] == (["text3"], 1)


def test_build_batches_mixed_token_sizes():
    """Test batch building with texts of varying token sizes."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=10,
    )

    # Mock tokenize to return varying token counts
    # Each call returns a list of token lists
    token_counts = [
        [[1] * 100],  # text1: 100 tokens
        [[1] * 200],  # text2: 200 tokens
        [[1] * 50],  # text3: 50 tokens
        [[1] * 150],  # text4: 150 tokens
    ]
    mock_tokenize = Mock(side_effect=token_counts)
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4"]
    batches = list(emb._build_batches(texts))

    # All texts should be included
    total_texts = sum(batch_size for _, batch_size in batches)
    assert total_texts == 4


def test_embed_with_batching():
    """Test _embed method uses batching correctly."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=2,
    )

    # Mock tokenize and embed
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    mock_embed_result = Mock()
    mock_embed_result.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_embed = Mock(return_value=mock_embed_result)
    emb._client.embed = mock_embed  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4"]
    result = emb._embed(texts, "document")

    # Should call embed twice (2 batches of 2 texts each)
    assert mock_embed.call_count == 2
    # Should return all 4 embeddings
    assert len(result) == 4


@pytest.mark.asyncio
async def test_aembed_with_batching():
    """Test _aembed method uses batching correctly."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=2,
    )

    # Mock tokenize and async embed
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    from unittest.mock import AsyncMock

    mock_embed_result = Mock()
    mock_embed_result.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_aembed = AsyncMock(return_value=mock_embed_result)
    emb._aclient.embed = mock_aembed  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4"]
    result = await emb._aembed(texts, "document")

    # Should call embed twice (2 batches of 2 texts each)
    assert mock_aembed.call_count == 2
    # Should return all 4 embeddings
    assert len(result) == 4


def test_embed_context_model_with_batching():
    """Test context model embedding uses batching correctly."""
    emb = VoyageEmbedding(
        model_name="voyage-context-3",
        voyage_api_key="NOT_A_VALID_KEY",
        embed_batch_size=2,
    )

    # Mock tokenize and contextualized_embed
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    mock_result_obj = Mock()
    mock_result_obj.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_embed_result = Mock()
    mock_embed_result.results = [mock_result_obj]
    mock_contextualized_embed = Mock(return_value=mock_embed_result)
    emb._client.contextualized_embed = mock_contextualized_embed  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4"]
    result = emb._embed(texts, "document")

    # Should call contextualized_embed twice
    assert mock_contextualized_embed.call_count == 2
    # Should return all 4 embeddings
    assert len(result) == 4


def test_automatic_batching_due_to_token_limits():
    """Test that batching happens automatically when token limits are exceeded."""
    emb = VoyageEmbedding(
        model_name="voyage-2",
        voyage_api_key="NOT_A_VALID_KEY",
        # Use default batch_size (1000) - batching should happen due to token limits
    )

    # Mock tokenize to return large token counts
    # Each text has 100k tokens, so with 320k limit for voyage-2,
    # we can fit max 3 texts per batch, but we have 5 texts
    mock_tokenize = Mock(return_value=[[1] * 100000])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    # Mock embed to return embeddings matching the batch size
    def mock_embed_side_effect(*args, **kwargs):
        # Get the batch of texts from the call
        texts_in_batch = args[0] if args else kwargs.get("texts", [])
        batch_size = len(texts_in_batch)
        mock_result = Mock()
        mock_result.embeddings = [[0.1, 0.2, 0.3] for _ in range(batch_size)]
        return mock_result

    mock_embed = Mock(side_effect=mock_embed_side_effect)
    emb._client.embed = mock_embed  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4", "text5"]
    result = emb._embed(texts, "document")

    # Verify all texts were embedded
    assert len(result) == 5

    # Verify multiple batches were created due to token limits
    # With 100k tokens per text and 320k limit:
    # Batch 1: text1, text2, text3 (300k tokens)
    # Batch 2: text4, text5 (200k tokens)
    assert mock_embed.call_count >= 2, (
        f"Expected at least 2 API calls, got {mock_embed.call_count}"
    )
