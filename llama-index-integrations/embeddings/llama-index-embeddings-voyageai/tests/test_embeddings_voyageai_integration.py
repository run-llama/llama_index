"""
Integration tests for VoyageAI embeddings with batching.

These tests require VOYAGE_API_KEY environment variable to be set.
Run with: pytest tests/test_embeddings_voyageai_integration.py -v
"""

import os

import pytest

from llama_index.embeddings.voyageai import VoyageEmbedding

# Skip all tests if VOYAGE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ, reason="VOYAGE_API_KEY not set"
)

MODEL = "voyage-3.5"
CONTEXT_MODEL = "voyage-context-3"


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_embedding_single_document(model: str):
    """Test embedding single document."""
    emb = VoyageEmbedding(model_name=model)
    text = "This is a test document."
    result = emb._get_text_embedding(text)

    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], float)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_embedding_multiple_documents(model: str):
    """Test embedding multiple documents."""
    emb = VoyageEmbedding(model_name=model, embed_batch_size=2)
    texts = ["Document 1", "Document 2", "Document 3"]
    result = emb._get_text_embeddings(texts)

    assert len(result) == 3
    assert all(isinstance(emb, list) for emb in result)
    # Verify embeddings are different
    assert result[0] != result[1]


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
async def test_async_embedding_multiple_documents(model: str):
    """Test async embedding multiple documents."""
    emb = VoyageEmbedding(model_name=model, embed_batch_size=2)
    texts = ["Document 1", "Document 2", "Document 3"]
    result = await emb._aget_text_embeddings(texts)

    assert len(result) == 3
    assert all(isinstance(emb, list) for emb in result)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_embedding_with_small_batch_size(model: str):
    """Test embedding with small batch size to verify batching works."""
    emb = VoyageEmbedding(model_name=model, embed_batch_size=2)
    texts = [f"Document {i}" for i in range(5)]
    result = emb._get_text_embeddings(texts)

    # Should successfully embed all documents despite small batch size
    assert len(result) == 5
    assert all(isinstance(emb, list) for emb in result)
    # Verify embeddings are unique
    assert result[0] != result[1]


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_embedding_empty_list(model: str):
    """Test embedding with empty list."""
    emb = VoyageEmbedding(model_name=model)
    texts = []
    result = emb._get_text_embeddings(texts)

    assert len(result) == 0
    assert isinstance(result, list)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_embedding_consistency(model: str):
    """Test that same text produces same embedding."""
    emb = VoyageEmbedding(model_name=model)
    text = "consistency test text"

    result1 = emb._get_text_embedding(text)
    result2 = emb._get_text_embedding(text)

    # Same text should produce identical embeddings
    assert result1 == result2


def test_embedding_with_output_dimension():
    """Test embedding with custom output dimension."""
    emb = VoyageEmbedding(
        model_name="voyage-3-large", output_dimension=512, embed_batch_size=10
    )
    texts = ["Test document"]
    result = emb._get_text_embeddings(texts)

    assert len(result) == 1
    assert len(result[0]) == 512


def test_context_model_embedding():
    """Test contextual embedding model."""
    emb = VoyageEmbedding(
        model_name="voyage-context-3", output_dimension=512, embed_batch_size=2
    )
    texts = ["Document 1", "Document 2", "Document 3"]
    result = emb._get_text_embeddings(texts)

    assert len(result) == 3
    assert all(len(emb) == 512 for emb in result)


async def test_context_model_async_embedding():
    """Test async contextual embedding model."""
    emb = VoyageEmbedding(
        model_name="voyage-context-3", output_dimension=512, embed_batch_size=2
    )
    texts = ["Document 1", "Document 2", "Document 3"]
    result = await emb._aget_text_embeddings(texts)

    assert len(result) == 3
    assert all(len(emb) == 512 for emb in result)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_automatic_batching_with_many_documents(model: str):
    """Test automatic batching with many documents."""
    emb = VoyageEmbedding(model_name=model, embed_batch_size=10)
    # Create 25 documents to ensure multiple batches
    texts = [f"Document number {i} with some content." for i in range(25)]
    result = emb._get_text_embeddings(texts)

    assert len(result) == 25
    assert all(isinstance(emb, list) for emb in result)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_batching_with_varying_text_lengths(model: str):
    """Test batching with texts of varying lengths."""
    emb = VoyageEmbedding(model_name=model, embed_batch_size=5)
    texts = [
        "Short.",
        "This is a medium length text with some more content.",
        "This is a much longer text that contains significantly more words and should consume more tokens than the previous texts. "
        * 3,
        "Another short one.",
        "Yet another long text with lots of repeated content. " * 5,
    ]
    result = emb._get_text_embeddings(texts)

    assert len(result) == 5
    assert all(isinstance(emb, list) for emb in result)


def test_query_vs_document_embeddings():
    """Test that query and document embeddings are different."""
    emb = VoyageEmbedding(model_name=MODEL)
    text = "test text"

    query_emb = emb._get_query_embedding(text)
    doc_emb = emb._get_text_embedding(text)

    # Query and document embeddings should be different
    assert query_emb != doc_emb
    assert len(query_emb) == len(doc_emb)


async def test_async_query_vs_document_embeddings():
    """Test async query and document embeddings are different."""
    emb = VoyageEmbedding(model_name=MODEL)
    text = "test text"

    query_emb = await emb._aget_query_embedding(text)
    doc_emb = await emb._aget_text_embedding(text)

    # Query and document embeddings should be different
    assert query_emb != doc_emb
    assert len(query_emb) == len(doc_emb)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_build_batches_with_real_tokenizer(model: str):
    """Test batch building with real tokenizer."""
    emb = VoyageEmbedding(model_name=model, embed_batch_size=10)
    texts = [
        "Short text.",
        "This is a much longer text with many more words.",
        "Another text.",
    ]

    batches = list(emb._build_batches(texts))

    # Verify all texts are included
    total_texts = sum(batch_size for _, batch_size in batches)
    assert total_texts == len(texts)

    # Verify each batch has texts
    for batch_texts, batch_size in batches:
        assert len(batch_texts) == batch_size
        assert batch_size > 0


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
@pytest.mark.slow
def test_automatic_batching_with_long_texts(model: str):
    """
    Test automatic batching with many texts that exceed token limits.

    This test is marked as slow because it processes many texts.
    The key is to have MANY texts whose combined tokens exceed the limit,
    not necessarily very long individual texts.
    """
    emb = VoyageEmbedding(model_name=model)

    # Create moderate-length text (not too long per text)
    # At roughly 4 chars per token, this is ~500 tokens per text
    text = "This is a document with some content for testing. " * 40

    # Create enough texts so combined tokens exceed the limit
    num_texts = 100

    texts = [f"{text} Document {i}." for i in range(num_texts)]

    # Count batches that will be created
    batches = list(emb._build_batches(texts))
    batch_count = len(batches)

    print(f"\nModel: {model}, Texts: {num_texts}, Batches: {batch_count}")

    # Verify multiple batches were created due to token limits
    assert batch_count >= 2, (
        f"Expected at least 2 batches, got {batch_count}. Model: {model}"
    )

    # Now actually embed them (this will take a while)
    result = emb._get_text_embeddings(texts)

    assert len(result) == num_texts
    assert all(isinstance(emb, list) for emb in result)
