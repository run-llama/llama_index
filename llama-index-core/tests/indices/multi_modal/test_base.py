"""Test MultiModalVectorStoreIndex with skip_embedding functionality."""

import pytest
from typing import List

from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import Document, ImageDocument, ImageNode, TextNode
from llama_index.core.storage.storage_context import StorageContext
from tests.indices.multi_modal.conftest import MockVectorStoreWithSkipEmbedding


def test_init_with_skip_embedding_false(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
) -> None:
    """Test that image_embed_model is properly initialized when skip_embedding=False."""
    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=False,
    )

    assert index._image_embed_model is not None
    assert isinstance(index._image_embed_model, MockMultiModalEmbedding)


def test_init_with_skip_embedding_true(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
) -> None:
    """Test that image_embed_model is None when skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )
    storage_context.add_vector_store(MockVectorStoreWithSkipEmbedding(), "image")

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert index._image_embed_model is None
    assert index._skip_embedding is True


def test_get_node_with_embedding_skip_text(
    mock_text_embed_model,
    mock_image_embed_model,
    text_nodes: List[TextNode],
) -> None:
    """Test that embeddings are cleared for TextNodes when skip_embedding=True."""
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=True,
    )

    result_nodes = index._get_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    assert len(result_nodes) == len(text_nodes)
    for node in result_nodes:
        assert node.embedding is None
        assert node.text == text_nodes[result_nodes.index(node)].text


def test_get_node_with_embedding_skip_image(
    mock_text_embed_model,
    mock_image_embed_model,
    image_nodes: List[ImageNode],
) -> None:
    """Test that both embedding and text_embedding are cleared for ImageNodes when skip_embedding=True."""
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=True,
    )

    result_nodes = index._get_node_with_embedding(
        nodes=image_nodes,
        show_progress=False,
        is_image=True,
    )

    assert len(result_nodes) == len(image_nodes)
    for node in result_nodes:
        assert isinstance(node, ImageNode)
        assert node.embedding is None
        assert node.text_embedding is None


def test_get_node_with_embedding_normal_text(
    mock_text_embed_model,
    mock_image_embed_model,
    text_nodes: List[TextNode],
) -> None:
    """Test that embeddings are generated normally for TextNodes when skip_embedding=False."""
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=False,
    )

    for node in text_nodes:
        node.embedding = None

    result_nodes = index._get_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    assert len(result_nodes) == len(text_nodes)
    for node in result_nodes:
        assert node.embedding is not None
        assert len(node.embedding) == 5  # mock embed dimension


@pytest.mark.asyncio
async def test_aget_node_with_embedding_skip_text(
    mock_text_embed_model,
    mock_image_embed_model,
    text_nodes: List[TextNode],
) -> None:
    """Test async: embeddings are cleared for TextNodes when skip_embedding=True."""
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=True,
    )

    result_nodes = await index._aget_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    assert len(result_nodes) == len(text_nodes)
    for node in result_nodes:
        assert node.embedding is None
        assert node.text == text_nodes[result_nodes.index(node)].text


@pytest.mark.asyncio
async def test_aget_node_with_embedding_skip_image(
    mock_text_embed_model,
    mock_image_embed_model,
    image_nodes: List[ImageNode],
) -> None:
    """Test async: both embedding and text_embedding are cleared for ImageNodes when skip_embedding=True."""
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=True,
    )

    result_nodes = await index._aget_node_with_embedding(
        nodes=image_nodes,
        show_progress=False,
        is_image=True,
    )

    assert len(result_nodes) == len(image_nodes)
    for node in result_nodes:
        assert isinstance(node, ImageNode)
        assert node.embedding is None
        assert node.text_embedding is None


@pytest.mark.asyncio
async def test_aget_node_with_embedding_normal_text(
    mock_text_embed_model,
    mock_image_embed_model,
    text_nodes: List[TextNode],
) -> None:
    """Test async: embeddings are generated normally for TextNodes when skip_embedding=False."""
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=False,
    )

    for node in text_nodes:
        node.embedding = None

    result_nodes = await index._aget_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    assert len(result_nodes) == len(text_nodes)
    for node in result_nodes:
        assert node.embedding is not None
        assert len(node.embedding) == 5  # mock embed dimension


def test_build_index_with_skip_embedding(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
    image_documents: List[ImageDocument],
) -> None:
    """Test building index with mixed documents when skip_embedding=True."""
    all_docs = documents + image_documents

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )
    storage_context.add_vector_store(MockVectorStoreWithSkipEmbedding(), "image")

    index = MultiModalVectorStoreIndex.from_documents(
        documents=all_docs,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    assert isinstance(index, MultiModalVectorStoreIndex)
    assert index._skip_embedding is True
    assert index._image_embed_model is None

    assert len(index.index_struct.nodes_dict) > 0

    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        assert node.embedding is None


def test_build_index_without_skip_embedding(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
    image_documents: List[ImageDocument],
) -> None:
    """Test building index with mixed documents when skip_embedding=False (normal operation)."""
    all_docs = documents + image_documents

    index = MultiModalVectorStoreIndex.from_documents(
        documents=all_docs,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        skip_embedding=False,
    )

    assert isinstance(index, MultiModalVectorStoreIndex)
    assert index._skip_embedding is False
    assert index._image_embed_model is not None

    assert len(index.index_struct.nodes_dict) > 0


def test_insert_with_skip_embedding(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
) -> None:
    """Test inserting documents when skip_embedding=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )
    storage_context.add_vector_store(MockVectorStoreWithSkipEmbedding(), "image")

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    initial_count = len(index.index_struct.nodes_dict)

    new_doc = Document(text="This is a new test.", id_="test_doc_3")
    index.insert(new_doc)

    assert len(index.index_struct.nodes_dict) > initial_count

    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        assert node.embedding is None


def test_insert_image_with_skip_embedding(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
) -> None:
    """Test that skip_embedding correctly handles mixed documents."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )
    storage_context.add_vector_store(MockVectorStoreWithSkipEmbedding(), "image")

    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    new_doc = Document(text="Additional text document.", id_="test_doc_3")
    index.insert(new_doc)

    for node_id in index.docstore.docs:
        node = index.docstore.get_node(node_id)
        assert node.embedding is None
        if isinstance(node, ImageNode):
            assert node.text_embedding is None


@pytest.mark.asyncio
async def test_async_add_nodes_with_skip_embedding(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
    image_documents: List[ImageDocument],
) -> None:
    """Test async adding nodes with skip_embedding=True."""
    all_docs = documents + image_documents

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )
    storage_context.add_vector_store(MockVectorStoreWithSkipEmbedding(), "image")

    index = MultiModalVectorStoreIndex.from_documents(
        documents=all_docs,
        embed_model=mock_text_embed_model,
        image_embed_model=mock_image_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
        use_async=True,
    )

    assert isinstance(index, MultiModalVectorStoreIndex)
    assert index._skip_embedding is True

    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        assert node.embedding is None
