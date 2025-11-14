"""Integration tests for MongoDB Atlas hybrid search functionality."""
import os
import time

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch, index

# Configuration
MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
COLLECTION_NAME = "llama_index_test_hybrid"
VECTOR_INDEX_NAME = "vector_index_hybrid"
TEXT_INDEX_NAME = "text_index_hybrid"
DIM = 8
TIMEOUT = 120
MAX_RETRY_ATTEMPTS = 8
RETRY_DELAY = 1.5


def _ensure_index_config(collection, index_name: str, index_type: str) -> None:
    """Ensure search index exists with proper configuration.
    
    Args:
        collection: MongoDB collection
        index_name: Name of the index
        index_type: Either 'vector' or 'text'
    """
    existing = {idx["name"]: idx for idx in collection.list_search_indexes()}
    
    if index_type == "vector":
        if index_name not in existing:
            index.create_vector_search_index(
                collection=collection,
                index_name=index_name,
                dimensions=DIM,
                path="embedding",
                similarity="cosine",
                filters=["metadata.tags"],
                wait_until_complete=TIMEOUT,
            )
        else:
            # Update to ensure tags filter exists
            index.update_vector_search_index(
                collection=collection,
                index_name=index_name,
                dimensions=DIM,
                path="embedding",
                similarity="cosine",
                filters=["metadata.tags"],
                wait_until_complete=TIMEOUT,
            )
    
    elif index_type == "text":
        text_mapping = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string"},
                    "metadata": {
                        "type": "document",
                        "fields": {"tags": {"type": "string"}},
                    },
                },
            }
        }
        
        if index_name not in existing:
            index.create_fulltext_search_index(
                collection=collection,
                index_name=index_name,
                field="text",
                field_type="string",
                wait_until_complete=TIMEOUT,
            )
            collection.update_search_index(name=index_name, definition=text_mapping)
        else:
            # Ensure metadata.tags mapping exists
            collection.update_search_index(name=index_name, definition=text_mapping)


def _wait_for_indexing(vector_store: MongoDBAtlasVectorSearch, timeout: int = 30) -> None:
    """Wait until documents are searchable via vector query.

    Args:
        vector_store: MongoDB vector store instance
        timeout: Maximum wait time in seconds

    Raises:
        TimeoutError: If documents not searchable within timeout
    """
    query = VectorStoreQuery(
        query_embedding=[0.9] * DIM,
        similarity_top_k=1,
        mode=VectorStoreQueryMode.DEFAULT
    )

    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            if vector_store.query(query).nodes:
                return
        except Exception:
            pass
        time.sleep(0.5)
    
    raise TimeoutError(f"Documents not searchable after {timeout}s")


def _query_with_retry(
    vector_store: MongoDBAtlasVectorSearch,
    query: VectorStoreQuery,
    min_results: int = 1,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    delay: float = RETRY_DELAY
):
    """Execute query with retry logic for index propagation delays.

    Args:
        vector_store: MongoDB vector store instance
        query: Query to execute
        min_results: Minimum expected results
        max_attempts: Maximum retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Query result

    Raises:
        AssertionError: If minimum results not achieved after all attempts
    """
    for attempt in range(max_attempts):
        result = vector_store.query(query)
        if len(result.nodes) >= min_results:
            return result
        time.sleep(delay if attempt < max_attempts // 2 else delay * 2)
    
    raise AssertionError(
        f"Query returned {len(result.nodes)} nodes after {max_attempts} attempts, "
        f"expected at least {min_results}"
    )


def _create_test_nodes() -> list[TextNode]:
    """Create test data nodes with predictable embeddings and metadata.

    Returns nodes with two groups:
    - Group A: High vector similarity ([0.85-0.9]), contains 'alpha' text, has tags
    - Group B: Low vector similarity ([0.1-0.2]), no 'alpha' text, varied tag presence
    """
    return [
        TextNode(
            text="alpha beta gamma",
            embedding=[0.9] * DIM,
            metadata={"group": "A", "tags": ["alpha", "news"]}
        ),
        TextNode(
            text="alpha beta",
            embedding=[0.85] * DIM,
            metadata={"group": "A", "tags": ["alpha"]}
        ),
        TextNode(
            text="delta epsilon",
            embedding=[0.1] * DIM,
            metadata={"group": "B", "tags": None}
        ),
        TextNode(
            text="zeta eta theta",
            embedding=[0.2] * DIM,
            metadata={"group": "B"}
        ),
    ]


@pytest.mark.skipif(
    MONGODB_URI is None,
    reason="Requires MONGODB_URI env variable for integration test",
)
class TestHybridIntegration:
    """Integration tests for hybrid search with vector and text indices."""

    @pytest.fixture(scope="class")
    def vector_store(self) -> MongoDBAtlasVectorSearch:
        """Create MongoDB Atlas vector store instance."""
        import pymongo
        client = pymongo.MongoClient(MONGODB_URI)
        return MongoDBAtlasVectorSearch(
            mongodb_client=client,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            vector_index_name=VECTOR_INDEX_NAME,
            fulltext_index_name=TEXT_INDEX_NAME,
        )

    @pytest.fixture(scope="class")
    def search_indexes(self, vector_store: MongoDBAtlasVectorSearch) -> None:
        """Ensure vector and text search indexes are configured."""
        collection = vector_store.collection
        _ensure_index_config(collection, VECTOR_INDEX_NAME, "vector")
        _ensure_index_config(collection, TEXT_INDEX_NAME, "text")

    @pytest.fixture(scope="class")
    def test_data(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        search_indexes: None
    ) -> None:
        """Seed test collection with predictable data."""
        vector_store.collection.delete_many({})
        vector_store.add(_create_test_nodes())
        _wait_for_indexing(vector_store)

    def test_hybrid_vector_bias(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        test_data: None
    ) -> None:
        """Test hybrid search with high alpha (vector-biased).

        With alpha=0.8, vector similarity should dominate. Query embedding close
        to group A should return group A documents first, despite text also matching.
        """
        query = VectorStoreQuery(
            query_embedding=[0.9] * DIM,
            query_str="alpha",
            similarity_top_k=3,
            hybrid_top_k=3,
            sparse_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.8,
        )

        result = _query_with_retry(vector_store, query, min_results=2)
        groups = [node.metadata.get("group") for node in result.nodes]

        assert groups[0] == "A", f"Expected group A first with vector bias, got {groups}"

    def test_hybrid_text_bias(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        test_data: None
    ) -> None:
        """Test hybrid search with low alpha (text-biased).

        With alpha=0.2, text relevance should dominate. Query text 'alpha' matches
        group A documents, so they should appear despite query embedding being closer
        to group B.
        """
        query = VectorStoreQuery(
            query_embedding=[0.2] * DIM,
            query_str="alpha",
            similarity_top_k=4,
            hybrid_top_k=4,
            sparse_top_k=4,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.2,
        )

        result = _query_with_retry(vector_store, query, min_results=2)
        groups = [node.metadata.get("group") for node in result.nodes]

        assert any(g == "A" for g in groups), f"Expected group A with text bias, got {groups}"

    def test_hybrid_filter_or_combination(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        test_data: None
    ) -> None:
        """Test hybrid search with OR filter combining IN and IS_EMPTY.
        
        Verifies that OR condition with IN and IS_EMPTY operators executes without
        error. Should match documents where tags contains 'alpha' OR tags is empty/missing.
        """
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value=["alpha"], operator=FilterOperator.IN),
                MetadataFilter(key="tags", value=None, operator=FilterOperator.IS_EMPTY),
            ],
            condition=FilterCondition.OR,
        )

        query = VectorStoreQuery(
            query_embedding=[0.85] * DIM,
            query_str="alpha",
            similarity_top_k=6,
            hybrid_top_k=6,
            sparse_top_k=6,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,
            filters=filters,
        )

        result = vector_store.query(query)
        assert result is not None, "Query with OR(IN, IS_EMPTY) should execute without error"

    def test_hybrid_filter_and_contradiction(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        test_data: None
    ) -> None:
        """Test hybrid search with logically contradictory AND filter.

        Verifies that AND condition combining IN and IS_EMPTY (which cannot both be
        true) executes without error and returns zero results.
        """
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value=["alpha"], operator=FilterOperator.IN),
                MetadataFilter(key="tags", value=None, operator=FilterOperator.IS_EMPTY),
            ],
            condition=FilterCondition.AND,
        )

        query = VectorStoreQuery(
            query_embedding=[0.9] * DIM,
            query_str="alpha",
            similarity_top_k=3,
            hybrid_top_k=3,
            sparse_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.7,
            filters=filters,
        )

        result = vector_store.query(query)
        assert result is not None, "Query should execute without error"
        assert len(result.nodes) == 0, (
            f"AND(IN, IS_EMPTY) should return 0 results, got {len(result.nodes)}"
        )

