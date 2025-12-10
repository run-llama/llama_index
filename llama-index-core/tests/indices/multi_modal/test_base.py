"""Test MultiModalVectorStoreIndex with skip_embedding functionality."""

import pytest
from typing import Any, List, Sequence

from llama_index.core import MockEmbedding
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    TextNode,
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore


class MockVectorStoreWithSkipEmbedding(SimpleVectorStore):
    """Mock vector store that supports skip_embedding by handling None embeddings."""

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index, handling None embeddings."""
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        for node in nodes:
            # Handle None embeddings (when skip_embedding=True)
            embedding = node.embedding if node.embedding is not None else []
            self.data.embedding_dict[node.node_id] = embedding
            self.data.text_id_to_ref_doc_id[node.node_id] = node.ref_doc_id or "None"

            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=False
            )
            metadata.pop("_node_content", None)
            self.data.metadata_dict[node.node_id] = metadata
        return [node.node_id for node in nodes]


@pytest.fixture()
def mock_text_embed_model():
    """Mock text embedding model with 5 dimensions."""
    return MockEmbedding(embed_dim=5)


@pytest.fixture()
def mock_image_embed_model():
    """Mock multimodal embedding model with 5 dimensions."""
    return MockMultiModalEmbedding(embed_dim=5)


@pytest.fixture()
def documents() -> List[Document]:
    """Sample documents for testing."""
    return [
        Document(text="Hello world.", id_="test_doc_1"),
        Document(text="This is a test.", id_="test_doc_2"),
    ]


@pytest.fixture()
def image_documents() -> List[ImageDocument]:
    """Sample image documents for testing."""
    # Base64 string for a 1×1 transparent PNG
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    return [
        ImageDocument(
            image=base64_str, metadata={"file_name": "test1.png"}, id_="test_img_1"
        ),
        ImageDocument(
            image=base64_str, metadata={"file_name": "test2.png"}, id_="test_img_2"
        ),
    ]


@pytest.fixture()
def text_nodes() -> List[TextNode]:
    """Sample text nodes with pre-existing embeddings."""
    return [
        TextNode(
            text="Hello world.", embedding=[1.0, 0.0, 0.0, 0.0, 0.0], id_="node_1"
        ),
        TextNode(
            text="This is a test.", embedding=[0.0, 1.0, 0.0, 0.0, 0.0], id_="node_2"
        ),
    ]


@pytest.fixture()
def image_nodes() -> List[ImageNode]:
    """Sample image nodes with pre-existing embeddings."""
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    return [
        ImageNode(
            image=base64_str,
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0],
            text_embedding=[0.0, 1.0, 0.0, 0.0, 0.0],
            id_="img_node_1",
        ),
        ImageNode(
            image=base64_str,
            embedding=[0.0, 0.0, 1.0, 0.0, 0.0],
            text_embedding=[0.0, 0.0, 1.0, 0.0, 0.0],
            id_="img_node_2",
        ),
    ]


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
    # Create storage context with mock vector stores that support skip_embedding
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

    # Call the method directly
    result_nodes = index._get_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    # Verify all embeddings are cleared
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

    # Call the method directly
    result_nodes = index._get_node_with_embedding(
        nodes=image_nodes,
        show_progress=False,
        is_image=True,
    )

    # Verify all embeddings are cleared
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

    # Clear existing embeddings to test generation
    for node in text_nodes:
        node.embedding = None

    # Call the method directly
    result_nodes = index._get_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    # Verify embeddings are generated
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

    # Call the async method directly
    result_nodes = await index._aget_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    # Verify all embeddings are cleared
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

    # Call the async method directly
    result_nodes = await index._aget_node_with_embedding(
        nodes=image_nodes,
        show_progress=False,
        is_image=True,
    )

    # Verify all embeddings are cleared
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

    # Clear existing embeddings to test generation
    for node in text_nodes:
        node.embedding = None

    # Call the async method directly
    result_nodes = await index._aget_node_with_embedding(
        nodes=text_nodes,
        show_progress=False,
        is_image=False,
    )

    # Verify embeddings are generated
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

    # Create storage context with mock vector stores that support skip_embedding
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

    # Verify nodes are in the index
    assert len(index.index_struct.nodes_dict) > 0

    # Verify nodes in docstore have no embeddings
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # Embeddings should be None when skip_embedding=True
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

    # Verify nodes are in the index
    assert len(index.index_struct.nodes_dict) > 0


def test_insert_with_skip_embedding(
    mock_text_embed_model,
    mock_image_embed_model,
    documents: List[Document],
) -> None:
    """Test inserting documents when skip_embedding=True."""
    # Create storage context with mock vector stores that support skip_embedding
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

    # Insert a new document
    new_doc = Document(text="This is a new test.", id_="test_doc_3")
    index.insert(new_doc)

    # Verify the document was added
    assert len(index.index_struct.nodes_dict) > initial_count

    # Verify new nodes have no embeddings
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
    # Create storage context with mock vector stores that support skip_embedding
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

    # Insert a new text document
    new_doc = Document(text="Additional text document.", id_="test_doc_3")
    index.insert(new_doc)

    # Verify all nodes continue to have no embeddings
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

    # Create storage context with mock vector stores that support skip_embedding
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

    # Verify nodes have no embeddings
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        assert node.embedding is None
