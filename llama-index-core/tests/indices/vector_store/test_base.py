"""Test VectorStoreIndex with skip_embedding functionality."""

import pytest
from typing import List

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import Document, ImageNode, IndexNode, TextNode
from llama_index.core.storage.storage_context import StorageContext
from tests.indices.multi_modal.conftest import MockVectorStoreWithSkipEmbedding


def test_skip_embedding_false_default(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that skip_embedding=False is default and generates embeddings."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    assert index._skip_embedding is False
    assert index._embed_model is not None
    assert len(index._vector_store.data.embedding_dict) > 0
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding is not None
        assert len(embedding) > 0


def test_skip_embedding_true_with_vector_store(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that skip_embedding=True prevents embedding generation."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert index._skip_embedding is True
    assert index._embed_model is not None
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_clears_existing_embeddings(
    mock_embed_model,
) -> None:
    """Test that skip_embedding=True clears pre-existing embeddings."""
    nodes = [
        TextNode(text="Hello world.", embedding=[1.0, 0.0, 0.0]),
        TextNode(text="This is a test.", embedding=[0.0, 1.0, 0.0]),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_text_nodes(
    mock_embed_model,
) -> None:
    """Test skip_embedding works with TextNode."""
    nodes = [
        TextNode(text="Hello world."),
        TextNode(text="This is a test."),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert len(index.index_struct.nodes_dict) == 2
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_index_node_raises_error(
    mock_embed_model,
) -> None:
    """Test that IndexNode raises error with skip_embedding=True."""
    nodes = [
        TextNode(text="Hello world."),
        IndexNode(index_id="some_index"),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    with pytest.raises(ValueError) as exc_info:
        VectorStoreIndex(
            nodes=nodes,
            embed_model=mock_embed_model,
            storage_context=storage_context,
            skip_embedding=True,
        )

    error_msg = str(exc_info.value)
    assert "IndexNode" in error_msg
    assert "skip_embedding=True" in error_msg
    assert "cannot be used" in error_msg


def test_skip_embedding_image_node_with_text(
    mock_embed_model,
) -> None:
    """Test that ImageNode with text works with skip_embedding=True."""
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

    nodes = [
        TextNode(text="Hello world."),
        ImageNode(image=base64_str, text="Image with text content."),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert len(index.index_struct.nodes_dict) == 2
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_image_node_with_image_url(
    mock_embed_model,
) -> None:
    """Test that ImageNode with image_url but no text gets filtered (no content)."""
    nodes = [
        TextNode(text="Hello world."),
        ImageNode(image_url="https://example.com/image.png"),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert len(index.index_struct.nodes_dict) == 1
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_image_node_without_text_raises_error(
    mock_embed_model,
) -> None:
    """Test that ImageNode without text/image_source raises error with skip_embedding=True."""
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

    nodes = [
        TextNode(text="Hello world."),
        ImageNode(image=base64_str),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    with pytest.raises(ValueError) as exc_info:
        VectorStoreIndex(
            nodes=nodes,
            embed_model=mock_embed_model,
            storage_context=storage_context,
            skip_embedding=True,
        )

    error_msg = str(exc_info.value)
    assert "text content" in error_msg
    assert "skip_embedding=True" in error_msg


def test_skip_embedding_empty_text_node_raises_error(
    mock_embed_model,
) -> None:
    """Test that empty TextNode raises error with skip_embedding=True."""
    nodes = [
        TextNode(text="Hello world."),
        TextNode(text=""),  # Empty text
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    with pytest.raises(ValueError) as exc_info:
        VectorStoreIndex(
            nodes=nodes,
            embed_model=mock_embed_model,
            storage_context=storage_context,
            skip_embedding=True,
        )

    error_msg = str(exc_info.value)
    assert "text content" in error_msg
    assert "skip_embedding=True" in error_msg


def test_skip_embedding_insert_nodes(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test insert_nodes with skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)

    index.insert(Document(text="New document text."))

    assert len(index.index_struct.nodes_dict) > initial_count

    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_insert_valid_document(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that valid documents are properly inserted with skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)

    index.insert(Document(text="Valid new document."))

    assert len(index.index_struct.nodes_dict) > initial_count
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


@pytest.mark.asyncio
async def test_skip_embedding_async_insert(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test async insert_nodes with skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)

    await index.ainsert(Document(text="Async new document."))

    assert len(index.index_struct.nodes_dict) > initial_count

    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


@pytest.mark.asyncio
async def test_skip_embedding_async_insert_nodes(
    mock_embed_model,
) -> None:
    """Test async insert_nodes with skip_embedding=True."""
    nodes = [
        TextNode(text="Initial text."),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    new_nodes = [
        TextNode(text="Async node 1."),
        TextNode(text="Async node 2."),
    ]
    await index.ainsert_nodes(new_nodes)

    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


@pytest.mark.asyncio
async def test_skip_embedding_async_insert_with_valid_nodes(
    mock_embed_model,
) -> None:
    """Test that async insert_nodes works with valid nodes and skip_embedding."""
    nodes = [
        TextNode(text="Initial text."),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)

    valid_nodes = [
        TextNode(text="New async node 1."),
        TextNode(text="New async node 2."),
    ]

    await index.ainsert_nodes(valid_nodes)

    assert len(index.index_struct.nodes_dict) > initial_count
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_whitespace_only_text_raises_error(
    mock_embed_model,
) -> None:
    """Test that whitespace-only text raises error with skip_embedding=True."""
    nodes = [TextNode(text="   \n\t  ")]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    with pytest.raises(ValueError) as exc_info:
        VectorStoreIndex(
            nodes=nodes,
            embed_model=mock_embed_model,
            storage_context=storage_context,
            skip_embedding=True,
        )

    error_msg = str(exc_info.value)
    assert "text content" in error_msg


def test_skip_embedding_multiple_invalid_nodes(
    mock_embed_model,
) -> None:
    """Test error includes node IDs when multiple nodes are invalid."""
    nodes = [
        TextNode(text="Valid content.", id_="node_1"),
        TextNode(text="", id_="node_2"),  # Invalid: empty
        TextNode(text="Valid content.", id_="node_3"),
        TextNode(text="", id_="node_4"),  # Invalid: empty
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    with pytest.raises(ValueError) as exc_info:
        VectorStoreIndex(
            nodes=nodes,
            embed_model=mock_embed_model,
            storage_context=storage_context,
            skip_embedding=True,
        )

    error_msg = str(exc_info.value)
    assert "node_2" in error_msg or "node_4" in error_msg


def test_skip_embedding_with_large_batch(
    mock_embed_model,
) -> None:
    """Test skip_embedding with large number of nodes."""
    nodes = [TextNode(text=f"Document {i}.") for i in range(100)]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert len(index.index_struct.nodes_dict) == 100
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_validation_happens_on_all_nodes(
    mock_embed_model,
) -> None:
    """Test that validation checks all nodes, not just filtered content nodes."""
    nodes = [
        TextNode(text="Valid content."),
        TextNode(text=""),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    with pytest.raises(ValueError) as exc_info:
        VectorStoreIndex(
            nodes=nodes,
            embed_model=mock_embed_model,
            storage_context=storage_context,
            skip_embedding=True,
        )

    error_msg = str(exc_info.value)
    assert "text content" in error_msg


def test_skip_embedding_retriever_creation(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that retriever can be created from skip_embedding index."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    retriever = index.as_retriever()
    assert retriever is not None
    assert retriever._index is index


def test_skip_embedding_delete_nodes(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test delete_nodes with skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)
    initial_vector_store_count = len(index._vector_store.data.embedding_dict)

    node_ids = list(index.index_struct.nodes_dict.values())[:1]

    index.delete_nodes(node_ids, delete_from_docstore=True)

    assert len(index.index_struct.nodes_dict) < initial_count
    assert len(index._vector_store.data.embedding_dict) < initial_vector_store_count


@pytest.mark.asyncio
async def test_skip_embedding_async_delete_nodes(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test adelete_nodes with skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)

    node_ids = list(index.index_struct.nodes_dict.values())[:1]

    await index.adelete_nodes(node_ids, delete_from_docstore=True)

    assert len(index.index_struct.nodes_dict) < initial_count


def test_skip_embedding_parameter_is_optional(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that skip_embedding parameter is optional."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    assert index._skip_embedding is False
    assert len(index._vector_store.data.embedding_dict) > 0
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding is not None and len(embedding) > 0


def test_skip_embedding_with_explicit_embed_model(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test skip_embedding=True with explicit embed_model provided."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert index._embed_model is not None
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_from_vector_store_initialization(
    mock_embed_model,
) -> None:
    """Test that from_vector_store initializes with skip_embedding parameter."""
    vector_store = MockVectorStoreWithSkipEmbedding()
    vector_store.stores_text = True

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=mock_embed_model,
        skip_embedding=True,
    )

    assert index._skip_embedding is True
    assert isinstance(index, VectorStoreIndex)


def test_skip_embedding_with_store_nodes_override_true(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test skip_embedding with store_nodes_override=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
        store_nodes_override=True,
    )

    assert index._store_nodes_override is True
    assert index._skip_embedding is True

    assert len(index.index_struct.nodes_dict) > 0
    assert len(index._vector_store.data.embedding_dict) > 0


def test_skip_embedding_with_store_nodes_override_false(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test skip_embedding with store_nodes_override=False (default)."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
        store_nodes_override=False,
    )

    assert index._store_nodes_override is False
    assert index._skip_embedding is True

    assert len(index.index_struct.nodes_dict) > 0


def test_skip_embedding_vector_store_receives_none_embeddings(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that vector store receives nodes with None embeddings."""
    vector_store = MockVectorStoreWithSkipEmbedding()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _ = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    for embedding in vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_text_nodes_stored_in_vector_store(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that nodes are properly stored in vector store with skip_embedding."""
    vector_store = MockVectorStoreWithSkipEmbedding()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert len(vector_store.data.embedding_dict) > 0
