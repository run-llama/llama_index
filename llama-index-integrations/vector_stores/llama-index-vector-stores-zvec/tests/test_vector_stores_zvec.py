import tempfile
import os
from typing import List
import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.zvec import ZvecVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in ZvecVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a simple embedding using basic character counts."""
    # Simple deterministic embedding based on ASCII values
    embedding = []
    for char in text[:50]:  # Take first 50 chars to avoid oversized embeddings
        embedding.append(float(ord(char) % 256) / 256.0)  # Normalize to 0-1 range
    # Pad to a fixed size (let's use 1536 as a common embedding size)
    embedding += [0.0] * (1536 - len(embedding))
    return embedding


@pytest.fixture()
def zvec_vector_store() -> ZvecVectorStore:
    """Create a temporary ZvecVectorStore for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "test.zvec")

    # Define sample metadata schema for testing
    metadata_schema = {
        "genre": "str",
        "pages": "int",
        "rating": "float",
        "in_stock": "bool",
    }

    vector_store = ZvecVectorStore(
        path=temp_path,
        collection_name="test_collection",
        collection_metadata=metadata_schema,
        embed_dim=1536,
    )

    yield vector_store

    # Close the collection connection before cleanup
    try:
        if hasattr(vector_store, "client") and vector_store.client is not None:
            # Try to explicitly sync/flush before closing if the method exists
            if hasattr(vector_store.client, "sync"):
                vector_store.client.sync()
            elif hasattr(vector_store.client, "flush"):
                vector_store.client.flush()

            # Close the collection properly
            vector_store.client.close()
    except Exception as e:
        # Log the error but don't fail the test
        print(f"Error closing zvec client: {e}")

    # Cleanup temp directory
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        # Log the error but don't fail the test
        print(f"Error removing temp directory: {e}")


@pytest.fixture()
def node_embeddings() -> List[TextNode]:
    """Return a list of TextNodes with embeddings."""
    return [
        TextNode(
            text="A story about mystery",
            id_="node-1",
            metadata={
                "genre": "Mystery",
                "pages": 10,
                "rating": 4.5,
                "in_stock": True,
            },
            embedding=text_to_embedding("A story about mystery"),
        ),
        TextNode(
            text="A comedy tale",
            id_="node-2",
            metadata={
                "genre": "Comedy",
                "pages": 5,
                "rating": 3.8,
                "in_stock": False,
            },
            embedding=text_to_embedding("A comedy tale"),
        ),
        TextNode(
            text="Thriller at midnight",
            id_="node-3",
            metadata={
                "genre": "Thriller",
                "pages": 20,
                "rating": 4.9,
                "in_stock": True,
            },
            embedding=text_to_embedding("Thriller at midnight"),
        ),
    ]


def test_initialization(zvec_vector_store: ZvecVectorStore) -> None:
    """Test basic initialization of ZvecVectorStore."""
    assert zvec_vector_store.client is not None
    assert zvec_vector_store.stores_text is True
    assert zvec_vector_store.flat_metadata is True


def test_add_and_query_nodes(
    zvec_vector_store: ZvecVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Test adding and querying nodes."""
    # Add nodes to the vector store
    node_ids = zvec_vector_store.add(node_embeddings)

    assert len(node_ids) == len(node_embeddings)

    # Query the vector store
    query_embedding = text_to_embedding("mystery story")
    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=2,
    )

    result = zvec_vector_store.query(query)

    assert result.nodes is not None
    assert len(result.nodes) <= 2
    assert result.similarities is not None
    assert len(result.similarities) == len(result.nodes)
    assert result.ids is not None
    assert len(result.ids) == len(result.nodes)


def test_delete_nodes(
    zvec_vector_store: ZvecVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Test deleting nodes from ZvecVectorStore."""
    # Add nodes to the vector store
    node_ids = zvec_vector_store.add(node_embeddings)

    # Verify nodes were added
    query_embedding = text_to_embedding("Thriller at midnight")
    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=1,
    )

    result = zvec_vector_store.query(query)
    assert len(result.nodes) >= 1

    # Delete a specific node
    zvec_vector_store.delete(ref_doc_id="node-3")

    # Query again - the deleted node should not appear
    result_after_delete = zvec_vector_store.query(query)

    # Check that the deleted node is not in the results
    if result_after_delete.nodes:
        node_ids_after = [node.node_id for node in result_after_delete.nodes]
        assert "node-3" not in node_ids_after


def test_metadata_filtering(
    zvec_vector_store: ZvecVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Test querying with metadata filters."""
    # Add nodes to the vector store
    zvec_vector_store.add(node_embeddings)

    # Query with a metadata filter
    query_embedding = text_to_embedding("story")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="genre", value="Mystery", operator="=="),
        ]
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=2,
        filters=filters,
    )

    result = zvec_vector_store.query(query)

    # Verify that results match the filter criteria
    if result.nodes:
        for node in result.nodes:
            assert node.metadata.get("genre") == "Mystery"


def test_sparse_vector_query() -> None:
    """Test sparse vector query functionality."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "test_sparse_query.zvec")

    try:
        # Create a vector store with sparse vector support
        sparse_store = ZvecVectorStore(
            path=temp_path,
            collection_name="test_sparse_query_collection",
            embed_dim=1536,
            support_sparse_vector=True,
        )

        assert sparse_store._support_sparse_vector is True

        # Create nodes with some content
        nodes = [
            TextNode(
                text="Machine learning algorithms improve performance",
                id_="sparse-node-1",
                embedding=text_to_embedding("Machine learning algorithms"),
            ),
            TextNode(
                text="Deep neural networks require training data",
                id_="sparse-node-2",
                embedding=text_to_embedding("Neural networks training"),
            ),
        ]

        # Add nodes to the store
        sparse_store.add(nodes)

        # Test sparse vector query
        query_str = "machine learning"
        query_embedding = text_to_embedding(query_str)

        query = VectorStoreQuery(
            query_str=query_str,
            query_embedding=query_embedding,
            similarity_top_k=2,
            mode=VectorStoreQueryMode.SPARSE,
        )

        # Perform the query
        result = sparse_store.query(query)

        # Verify results
        assert result.nodes is not None
        assert len(result.nodes) <= 2
        assert result.similarities is not None
        assert len(result.similarities) == len(result.nodes)
        assert result.ids is not None
        assert len(result.ids) == len(result.nodes)

        # Test hybrid query (dense + sparse)
        hybrid_query = VectorStoreQuery(
            query_str=query_str,
            query_embedding=query_embedding,
            similarity_top_k=2,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,  # Balance between dense and sparse
        )

        hybrid_result = sparse_store.query(hybrid_query)

        # Verify hybrid results
        assert hybrid_result.nodes is not None
        assert len(hybrid_result.nodes) <= 2

        # Clean up
        try:
            sparse_store.client.close()
        except Exception:
            pass

    except ImportError:
        # If dashtext is not available, skip this test
        pass

    # Clean up temp directory
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def test_invalid_initialization() -> None:
    """Test that initialization fails with invalid parameters."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "test_invalid.zvec")

    # Test with missing path
    with pytest.raises(ValueError):
        ZvecVectorStore(
            path=None,  # type: ignore
            collection_name="test",
            embed_dim=1536,
        )

    # Test with missing collection_name
    with pytest.raises(ValueError):
        ZvecVectorStore(
            path=temp_path,
            collection_name=None,  # type: ignore
            embed_dim=1536,
        )

    # Test with missing embed_dim
    with pytest.raises(ValueError):
        ZvecVectorStore(
            path=temp_path,
            collection_name="test",
            embed_dim=None,  # type: ignore
        )

    # Clean up temp directory
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error removing invalid init temp directory: {e}")


def test_unsupported_metadata_type() -> None:
    """Test that initialization fails with unsupported metadata type."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "test_meta.zvec")

    with pytest.raises(ValueError):
        ZvecVectorStore(
            path=temp_path,
            collection_name="test",
            collection_metadata={"invalid_field": "unsupported_type"},  # type: ignore
            embed_dim=1536,
        )

    # Clean up temp directory
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error removing metadata type test temp directory: {e}")
