"""Tests for MultiModalVectorIndexRetriever with skip_embedding functionality."""

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from tests.indices.multi_modal.conftest import MockVectorStoreWithSkipEmbedding


def test_retriever_with_skip_embedding_true(
    mock_text_embed_model,
):
    """Test that retriever properly handles skip_embedding=True scenario."""
    documents = [
        Document(text="This is a test document about AI."),
    ]

    # Create storage context with mock vector stores to avoid embedding requirement
    storage_context = StorageContext.from_defaults()
    storage_context.add_vector_store(
        MockVectorStoreWithSkipEmbedding(), namespace="default"
    )
    storage_context.add_vector_store(
        MockVectorStoreWithSkipEmbedding(), namespace="image"
    )

    # Create index with skip_embedding=True and image_embed_model=None
    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=None,
        skip_embedding=True,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(similarity_top_k=2)

    assert retriever is not None
    assert retriever._index is not None
    # When skip_embedding=True, image_embed_model should be None
    assert retriever._image_embed_model is None
    assert retriever._embed_model is not None


def test_retriever_with_skip_embedding_false(
    mock_text_embed_model,
    mock_image_embed_model,
):
    """Test that retriever is created with skip_embedding=False and both embed models exist."""
    documents = [
        Document(text="This is a test document about AI."),
        Document(text="Another test document about machine learning."),
    ]

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=False,
    )

    retriever = index.as_retriever(similarity_top_k=3)

    assert retriever is not None
    assert retriever._image_embed_model is not None
    assert retriever._embed_model is not None


def test_retriever_embed_model_none_check(
    mock_text_embed_model,
):
    """Test retriever behavior when image_embed_model is None (skip_embedding scenario)."""
    documents = [
        Document(text="This is a test document about AI."),
    ]

    # Create storage context with mock vector stores to avoid embedding requirement
    storage_context = StorageContext.from_defaults()
    storage_context.add_vector_store(
        MockVectorStoreWithSkipEmbedding(), namespace="default"
    )
    storage_context.add_vector_store(
        MockVectorStoreWithSkipEmbedding(), namespace="image"
    )

    # Use skip_embedding=True when image_embed_model is None
    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=None,
        skip_embedding=True,
        storage_context=storage_context,
    )

    retriever = index.as_retriever()

    # When skip_embedding=True, image_embed_model should be None
    # This prevents the retriever from generating embeddings for image queries
    assert retriever._image_embed_model is None

    # Embed model should still be available for text search
    assert retriever._embed_model is not None


def test_retriever_similarity_top_k_property(
    mock_text_embed_model,
    mock_image_embed_model,
):
    """Test that similarity_top_k property can be get and set."""
    documents = [
        Document(text="This is a test document about AI."),
    ]

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=5)

    assert retriever.similarity_top_k == 5

    retriever.similarity_top_k = 10
    assert retriever.similarity_top_k == 10


def test_retriever_index_access(
    mock_text_embed_model,
    mock_image_embed_model,
):
    """Test that retriever properly accesses the index."""
    documents = [
        Document(text="This is a test document about AI."),
    ]

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
    )

    retriever = index.as_retriever()

    # Verify retriever has reference to index
    assert retriever._index is index
    assert retriever._index is not None


def test_retriever_preserves_embed_models_from_index(
    mock_text_embed_model,
    mock_image_embed_model,
):
    """Test that retriever preserves embed models from the index."""
    documents = [
        Document(text="This is a test document about AI."),
    ]

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=False,
    )

    retriever = index.as_retriever()

    # Both embed models should be available from the retriever
    assert retriever._embed_model is not None
    assert retriever._image_embed_model is not None
    # They should match the index models
    assert retriever._embed_model is index._embed_model
