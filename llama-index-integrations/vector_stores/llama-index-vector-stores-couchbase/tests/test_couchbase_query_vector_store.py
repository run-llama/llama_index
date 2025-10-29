"""Test Couchbase Query Vector Store functionality using GSI."""

from __future__ import annotations
import os
import json
from typing import Any, List
from datetime import timedelta

import pytest
import time

from llama_index.core.schema import MetadataMode, TextNode, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.vector_stores.couchbase import CouchbaseQueryVectorStore
from llama_index.vector_stores.couchbase.base import QueryVectorSearchType
from llama_index.vector_stores.couchbase.base import QueryVectorSearchSimilarity
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex

from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.logic.options import KnownConfigProfiles
from couchbase.options import QueryOptions

CONNECTION_STRING = os.getenv("COUCHBASE_CONNECTION_STRING", "")
BUCKET_NAME = os.getenv("COUCHBASE_BUCKET_NAME", "")
SCOPE_NAME = os.getenv("COUCHBASE_SCOPE_NAME", "")
COLLECTION_NAME = os.getenv("COUCHBASE_COLLECTION_NAME", "")
USERNAME = os.getenv("COUCHBASE_USERNAME", "")
PASSWORD = os.getenv("COUCHBASE_PASSWORD", "")
INDEX_NAME = os.getenv("COUCHBASE_INDEX_NAME", "test_vector_index")
SLEEP_DURATION = 5  # Increased for GSI indexing
EMBEDDING_DIMENSION = 1536


def set_all_env_vars() -> bool:
    """Check if all required environment variables are set."""
    return all(
        [
            CONNECTION_STRING,
            BUCKET_NAME,
            SCOPE_NAME,
            COLLECTION_NAME,
            USERNAME,
            PASSWORD,
        ]
    )


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    # Pad or trim the list to make it of length EMBEDDING_DIMENSION
    return ascii_values[:EMBEDDING_DIMENSION] + [0.0] * (
        EMBEDDING_DIMENSION - len(ascii_values)
    )


def get_cluster() -> Any:
    """Get a couchbase cluster object."""
    auth = PasswordAuthenticator(USERNAME, PASSWORD)
    options = ClusterOptions(authenticator=auth)
    options.apply_profile(KnownConfigProfiles.WanDevelopment)
    connect_string = CONNECTION_STRING
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@pytest.fixture()
def cluster() -> Cluster:
    """Get a couchbase cluster object."""
    return get_cluster()


def delete_documents(
    client: Any, bucket_name: str, scope_name: str, collection_name: str
) -> None:
    """Delete all the documents in the collection."""
    query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
    client.query(query).execute()


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    """Return a list of TextNodes with embeddings."""
    return [
        TextNode(
            text="foo",
            id_="12c70eed-5779-4008-aba0-596e003f6443",
            metadata={
                "genre": "Mystery",
                "pages": 10,
                "rating": 4.5,
            },
            embedding=text_to_embedding("foo"),
        ),
        TextNode(
            text="bar",
            id_="f7d81cb3-bb42-47e6-96f5-17db6860cd11",
            metadata={
                "genre": "Comedy",
                "pages": 5,
                "rating": 3.2,
            },
            embedding=text_to_embedding("bar"),
        ),
        TextNode(
            text="baz",
            id_="469e9537-7bc5-4669-9ff6-baa0ed086236",
            metadata={
                "genre": "Thriller",
                "pages": 20,
                "rating": 4.8,
            },
            embedding=text_to_embedding("baz"),
        ),
    ]


def create_scope_and_collection(
    cluster: Cluster, bucket_name: str, scope_name: str, collection_name: str
) -> None:
    """Create scope and collection if they don't exist."""
    try:
        from couchbase.exceptions import (
            ScopeAlreadyExistsException,
            CollectionAlreadyExistsException,
            QueryIndexAlreadyExistsException,
        )

        bucket = cluster.bucket(bucket_name)

        # Create scope if it doesn't exist
        try:
            bucket.collections().create_scope(scope_name=scope_name)
        except ScopeAlreadyExistsException:
            pass

        # Create collection if it doesn't exist
        try:
            bucket.collections().create_collection(
                collection_name=collection_name, scope_name=scope_name
            )
        except CollectionAlreadyExistsException:
            pass

        try:
            bucket.scope(scope_name).collection(
                collection_name
            ).query_indexes().create_primary_index()
        except QueryIndexAlreadyExistsException:
            pass

    except Exception as e:
        # Log the error but don't fail - collection might already exist
        pass


def create_vector_index(
    cluster: Any,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    index_name: str,
    embedding_key: str = "embedding",
) -> None:
    """Create GSI vector index for the collection."""
    try:
        from couchbase.options import QueryOptions

        bucket = cluster.bucket(bucket_name)
        scope = bucket.scope(scope_name)

        # Check if index already exists
        try:
            query = f"SELECT name FROM system:indexes WHERE keyspace_id = '{collection_name}' AND name = '{index_name}'"
            result = scope.query(query).execute()
            if len(list(result.rows())) > 0:
                return  # Index already exists
        except Exception:
            pass

        # Index creation options
        with_opts = json.dumps(
            {
                "dimension": EMBEDDING_DIMENSION,
                "description": "IVF1024,PQ32x8",
                "similarity": "cosine",
            }
        )

        collection = scope.collection(collection_name)

        docs = {}
        for i in range(2000):
            docs[f"large_batch_{i}"] = {
                "text": f"document_{i}",
                "embedding": text_to_embedding(f"document_{i}"),
                "metadata": {
                    "batch_id": "large",
                    "doc_num": i,
                },
            }

        result = collection.insert_multi(docs)
        if not result.all_ok:
            raise Exception(f"Error inserting documents: {result.exceptions}")

        # Create vector index
        create_index_query = f"""
        CREATE INDEX {index_name}
        ON `{bucket_name}`.`{scope_name}`.`{collection_name}` ({embedding_key} VECTOR)
        USING GSI WITH {with_opts}
        """
        result = scope.query(
            create_index_query, QueryOptions(timeout=timedelta(seconds=300))
        ).execute()
        time.sleep(15)
        # raise Exception("Stop here")

        # Wait for index to be ready

    except Exception:
        raise


def drop_vector_index(
    cluster: Any,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    index_name: str,
) -> None:
    """Drop the GSI vector index."""
    try:
        from couchbase.options import QueryOptions

        bucket = cluster.bucket(bucket_name)
        scope = bucket.scope(scope_name)

        drop_index_query = f"DROP INDEX `{index_name}` on `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        scope.query(
            drop_index_query, QueryOptions(timeout=timedelta(seconds=60))
        ).execute()

    except Exception as e:
        # Index might not exist or already dropped
        pass


@pytest.mark.skipif(
    not set_all_env_vars(), reason="missing Couchbase environment variables"
)
class TestCouchbaseQueryVectorStore:
    @classmethod
    def setup_class(cls) -> None:
        """Set up test class with vector index creation."""
        cls.cluster = get_cluster()

        # Create scope and collection if they don't exist
        create_scope_and_collection(
            cls.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME
        )

        # Create vector index for testing
        create_vector_index(
            cls.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, INDEX_NAME
        )

    @classmethod
    def teardown_class(cls) -> None:
        """Clean up after all tests."""
        try:
            # Drop the vector index
            drop_vector_index(
                cls.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, INDEX_NAME
            )
            delete_documents(cls.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
        except Exception:
            pass

    def setup_method(self) -> None:
        """Set up each test method."""
        # Delete all the documents in the collection
        delete_documents(self.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
        self.vector_store = CouchbaseQueryVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.DOT,
            nprobes=50,
        )

    def test_initialization_default_params(self) -> None:
        """Test initialization with default parameters."""
        vector_store = CouchbaseQueryVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
            nprobes=50,
        )

        assert vector_store._search_type == QueryVectorSearchType.ANN
        assert vector_store._similarity == QueryVectorSearchSimilarity.COSINE
        assert vector_store._nprobes == 50
        assert vector_store._text_key == "text"
        assert vector_store._embedding_key == "embedding"
        assert vector_store._metadata_key == "metadata"

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        custom_timeout = timedelta(seconds=120)
        vector_store = CouchbaseQueryVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            search_type=QueryVectorSearchType.KNN,
            similarity="euclidean",
            text_key="content",
            embedding_key="vector",
            metadata_key="meta",
            query_options=QueryOptions(timeout=custom_timeout),
        )

        assert vector_store._search_type == QueryVectorSearchType.KNN
        assert vector_store._similarity == QueryVectorSearchSimilarity.EUCLIDEAN
        assert vector_store._text_key == "content"
        assert vector_store._embedding_key == "vector"
        assert vector_store._metadata_key == "meta"
        assert vector_store._query_options["timeout"] == custom_timeout

    def test_initialization_with_string_search_type(self) -> None:
        """Test initialization with string search type."""
        vector_store = CouchbaseQueryVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            search_type="KNN",
            similarity="EUCLIDEAN",
        )

        assert vector_store._search_type == QueryVectorSearchType.KNN
        assert vector_store._similarity == QueryVectorSearchSimilarity.EUCLIDEAN
        assert vector_store._nprobes is None

    def test_add_documents(self, node_embeddings: List[TextNode]) -> None:
        """Test adding documents to Couchbase query vector store."""
        input_doc_ids = [node_embedding.id_ for node_embedding in node_embeddings]
        # Add nodes to the couchbase vector store
        doc_ids = self.vector_store.add(node_embeddings)

        # Ensure that all nodes are returned & they are the same as input
        assert len(doc_ids) == len(node_embeddings)
        for doc_id in doc_ids:
            assert doc_id in input_doc_ids

    def test_ann_search(self, node_embeddings: List[TextNode]) -> None:
        """Test ANN vector search functionality."""
        # Add nodes to the couchbase vector store
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # ANN similarity search
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"), similarity_top_k=1
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
        )
        assert result.similarities is not None

    def test_knn_search(self, node_embeddings: List[TextNode]) -> None:
        """Test KNN vector search functionality."""
        # Create a KNN vector store
        knn_vector_store = CouchbaseQueryVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            search_type=QueryVectorSearchType.KNN,
            similarity=QueryVectorSearchSimilarity.L2,
            nprobes=50,
        )

        # Add nodes to the couchbase vector store
        knn_vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # KNN similarity search
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"), similarity_top_k=1
        )

        result = knn_vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
        )
        assert result.similarities is not None

    def test_search_with_filters(self, node_embeddings: List[TextNode]) -> None:
        """Test vector search with metadata filters."""
        # Add nodes to the couchbase vector store
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # Test equality filter
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="genre", value="Thriller", operator=FilterOperator.EQ
                    ),
                ]
            ),
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert result.nodes[0].metadata.get("genre") == "Thriller"

    def test_search_with_numeric_filters(self, node_embeddings: List[TextNode]) -> None:
        """Test vector search with numeric metadata filters."""
        # Add nodes to the couchbase vector store
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # Test greater than filter
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="pages", value=10, operator=FilterOperator.GT),
                ]
            ),
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert result.nodes[0].metadata.get("pages") == 20

        # Test less than or equal filter
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("bar"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="pages", value=10, operator=FilterOperator.LTE),
                ]
            ),
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 2
        for node in result.nodes:
            assert node.metadata.get("pages") <= 10

    def test_search_with_combined_filters(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """Test vector search with multiple combined filters."""
        # Add nodes to the couchbase vector store
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # Test combined filters with AND condition
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="genre", value="Thriller", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(key="rating", value=4.0, operator=FilterOperator.GT),
                ],
                condition="and",
            ),
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert result.nodes[0].metadata.get("genre") == "Thriller"
        assert result.nodes[0].metadata.get("rating") > 4.0

    def test_delete_document(self) -> None:
        """Test delete document from Couchbase query vector store."""
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Add a document to the vector store
        VectorStoreIndex.from_documents(
            [
                Document(
                    text="hello world",
                    metadata={"name": "John Doe", "age": 30, "city": "New York"},
                ),
            ],
            storage_context=storage_context,
        )

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # Search for the document
        search_embedding = OpenAIEmbedding().get_text_embedding("hello world")
        q = VectorStoreQuery(
            query_embedding=search_embedding,
            similarity_top_k=1,
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1

        # Get the document ID to delete
        ref_doc_id_to_delete = result.nodes[0].ref_doc_id

        # Delete the document
        self.vector_store.delete(ref_doc_id=ref_doc_id_to_delete)

        # Wait for the deletion to be processed
        time.sleep(SLEEP_DURATION)

        # Ensure that no results are returned
        result = self.vector_store.query(q)
        assert len(result.nodes) == 0

    def test_empty_query_embedding_error(self) -> None:
        """Test that empty query embedding raises ValueError."""
        q = VectorStoreQuery(
            query_embedding=None,
            similarity_top_k=1,
        )

        with pytest.raises(ValueError, match="Query embedding must not be empty"):
            self.vector_store.query(q)

    def test_different_similarity_metrics(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """Test different similarity metrics."""
        similarity_metrics = [
            QueryVectorSearchSimilarity.COSINE,
            QueryVectorSearchSimilarity.EUCLIDEAN,
            QueryVectorSearchSimilarity.DOT,
        ]

        for metric in similarity_metrics:
            # Create vector store with specific similarity metric
            vector_store = CouchbaseQueryVectorStore(
                cluster=self.cluster,
                bucket_name=BUCKET_NAME,
                scope_name=SCOPE_NAME,
                collection_name=COLLECTION_NAME,
                similarity=metric,
                search_type=QueryVectorSearchType.ANN,
                nprobes=50,
            )

            # Add nodes to the vector store
            vector_store.add(node_embeddings)

            # Wait for indexing
            time.sleep(SLEEP_DURATION)

            # Test search
            q = VectorStoreQuery(
                query_embedding=text_to_embedding("foo"),
                similarity_top_k=1,
            )

            result = vector_store.query(q)
            assert result.nodes is not None and len(result.nodes) == 1
            assert result.similarities is not None

    def test_custom_field_names(self) -> None:
        """Test vector store with custom field names."""
        custom_vector_store = CouchbaseQueryVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
            nprobes=50,
            text_key="content",
            embedding_key="vector",
            metadata_key="meta",
        )

        # Create a test node with custom field mapping
        test_node = TextNode(
            text="custom field test",
            id_="custom-test-id",
            metadata={"category": "test"},
            embedding=text_to_embedding("custom field test"),
        )

        # Add the node
        doc_ids = custom_vector_store.add([test_node])
        assert len(doc_ids) == 1

        # Wait for indexing
        time.sleep(SLEEP_DURATION)

        # Search for the document
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("custom field test"),
            similarity_top_k=1,
        )

        result = custom_vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == "custom field test"
        )

    def test_batch_insert(self, node_embeddings: List[TextNode]) -> None:
        """Test batch insert with custom batch size."""
        # Test with small batch size
        doc_ids = self.vector_store.add(node_embeddings, batch_size=2)
        assert len(doc_ids) == len(node_embeddings)

        # Wait for indexing
        time.sleep(SLEEP_DURATION)

        # Verify all documents are searchable
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 3

    def test_vector_index_utilization(self, node_embeddings: List[TextNode]) -> None:
        """Test that vector search actually utilizes the GSI vector index."""
        # Add nodes to the vector store
        self.vector_store.add(node_embeddings)

        # Wait for GSI indexing
        time.sleep(SLEEP_DURATION)

        # Test that we can perform vector search (this implicitly tests index usage)
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=2,
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 2
        assert result.similarities is not None
        assert len(result.similarities) == 2

    def test_vector_search_relevance(self, node_embeddings: List[TextNode]) -> None:
        """Test that vector search returns relevant results."""
        # Add nodes to the vector store
        self.vector_store.add(node_embeddings)

        # Wait for GSI indexing
        time.sleep(SLEEP_DURATION)

        # Search for "foo" - should return "foo" document with best score
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 3

        # The first result should be the most similar (lowest distance for dot product)
        assert result.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

        # Verify scores are ordered (ascending for distance-based similarity)
        scores = result.similarities
        print(f"scores: {scores}")
        assert scores[0] <= scores[1]
        assert scores[1] <= scores[2]

    def test_large_batch_processing(self) -> None:
        """Test handling of larger document batches."""
        # Create a larger batch of documents
        large_batch = []
        for i in range(2000):
            node = TextNode(
                text=f"document_{i}",
                id_=f"large_batch_{i}",
                metadata={"batch_id": "large", "doc_num": i},
                embedding=text_to_embedding(f"document_{i}"),
            )
            large_batch.append(node)

        # Add the large batch
        doc_ids = self.vector_store.add(large_batch, batch_size=10)
        assert len(doc_ids) == len(large_batch)

        # Wait for indexing
        time.sleep(SLEEP_DURATION * 2)  # Extra time for larger batch

        # Test search works with larger dataset
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("document_25"),
            similarity_top_k=5,
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 5
