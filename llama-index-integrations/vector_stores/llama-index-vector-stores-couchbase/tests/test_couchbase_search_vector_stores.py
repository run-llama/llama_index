"""Test Couchbase Vector Search functionality."""

from __future__ import annotations
import os
from typing import Any, List

import pytest
import time
import json

from llama_index.core.schema import MetadataMode, TextNode, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
)
from llama_index.vector_stores.couchbase import (
    CouchbaseVectorStore,
    CouchbaseSearchVectorStore,
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from couchbase.cluster import Cluster
from couchbase.management.logic.search_index_logic import SearchIndex
from couchbase.exceptions import SearchIndexNotFoundException


CONNECTION_STRING = os.getenv("COUCHBASE_CONNECTION_STRING", "")
BUCKET_NAME = os.getenv("COUCHBASE_BUCKET_NAME", "")
SCOPE_NAME = os.getenv("COUCHBASE_SCOPE_NAME", "")
COLLECTION_NAME = os.getenv("COUCHBASE_COLLECTION_NAME", "")
USERNAME = os.getenv("COUCHBASE_USERNAME", "")
PASSWORD = os.getenv("COUCHBASE_PASSWORD", "")
INDEX_NAME = os.getenv("COUCHBASE_INDEX_NAME", "")
SLEEP_DURATION = 1
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
            INDEX_NAME,
        ]
    )


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    # Pad or trim the list to make it of length ADA_TOKEN_COUNT
    return ascii_values[:EMBEDDING_DIMENSION] + [0.0] * (
        EMBEDDING_DIMENSION - len(ascii_values)
    )


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
    cluster: Cluster,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    index_name: str,
) -> None:
    """Create vector index if it doesn't exist."""
    bucket = cluster.bucket(BUCKET_NAME)
    scope = bucket.scope(SCOPE_NAME)
    index_definition = load_json_file(f"{os.path.dirname(__file__)}/vector_index.json")

    sim = scope.search_indexes()
    try:
        sim.get_index(index_name=index_definition["name"])
    except SearchIndexNotFoundException as e:
        type = index_definition["params"]["mapping"]["types"][
            "____scope.collection_____"
        ]
        del index_definition["params"]["mapping"]["types"]["____scope.collection_____"]
        index_definition["params"]["mapping"]["types"][
            f"{SCOPE_NAME}.{COLLECTION_NAME}"
        ] = type
        search_index = SearchIndex(
            name=index_definition["name"],
            source_name=BUCKET_NAME,
            source_type=index_definition["sourceType"],
            params=index_definition["params"],
            plan_params=index_definition["planParams"],
        )
        sim.upsert_index(search_index)

    #  Wait for the index to be ready
    max_retries = 10
    retry_interval = 2  # seconds
    for attempt in range(max_retries):
        try:
            # Check if index exists and is ready by getting document count
            sim.get_indexed_documents_count(index_definition["name"])
            # If we can get the count, the index is ready
            break
        except Exception as e:
            pass

        time.sleep(retry_interval)
        if attempt == max_retries - 1:
            pytest.skip(
                f"Index {index_definition['name']} not ready after {max_retries} attempts"
            )


@pytest.fixture(scope="session")
def cluster() -> Cluster:
    """Get a couchbase cluster object."""
    from datetime import timedelta

    from couchbase.auth import PasswordAuthenticator
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions

    auth = PasswordAuthenticator(USERNAME, PASSWORD)
    options = ClusterOptions(auth)
    connect_string = CONNECTION_STRING
    cluster = Cluster(connect_string, options)
    bucket = cluster.bucket(BUCKET_NAME)
    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))
    create_scope_and_collection(cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
    create_vector_index(cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, INDEX_NAME)

    yield cluster
    bucket.collections().drop_scope(SCOPE_NAME)
    cluster.close()


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
            },
            embedding=text_to_embedding("foo"),
        ),
        TextNode(
            text="bar",
            id_="f7d81cb3-bb42-47e6-96f5-17db6860cd11",
            metadata={
                "genre": "Comedy",
                "pages": 5,
            },
            embedding=text_to_embedding("bar"),
        ),
        TextNode(
            text="cake",
            id_="469e9537-7bc5-4669-9ff6-baa0ed086236",
            metadata={
                "genre": "Thriller",
                "pages": 20,
            },
            embedding=text_to_embedding("cake"),
        ),
    ]


@pytest.mark.skipif(
    not set_all_env_vars(), reason="missing Couchbase environment variables"
)
class TestCouchbaseSearchVectorStore:
    @pytest.fixture()
    def vector_store(self, cluster: Cluster) -> CouchbaseSearchVectorStore:
        yield CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
        )
        delete_documents(cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)

    def test_add_documents(
        self, vector_store: CouchbaseSearchVectorStore, node_embeddings: List[TextNode]
    ) -> None:
        """Test adding documents to Couchbase vector store."""
        input_doc_ids = [node_embedding.id_ for node_embedding in node_embeddings]
        # Add nodes to the couchbase vector
        doc_ids = vector_store.add(node_embeddings)

        # Ensure that all nodes are returned & they are the same as input
        assert len(doc_ids) == len(node_embeddings)
        for doc_id in doc_ids:
            assert doc_id in input_doc_ids

    def test_search(
        self, vector_store: CouchbaseSearchVectorStore, node_embeddings: List[TextNode]
    ) -> None:
        """Test end to end Couchbase vector search."""
        # Add nodes to the couchbase vector
        vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # similarity search
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"), similarity_top_k=1
        )

        result = vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
        )
        assert result.similarities is not None

    def test_delete_doc(self, vector_store: CouchbaseSearchVectorStore) -> None:
        """Test delete document from Couchbase vector store."""
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Add nodes to the couchbase vector
        store_index = VectorStoreIndex.from_documents(
            [
                Document(
                    text="hello",
                    metadata={"name": "John Doe", "age": 30, "city": "New"},
                ),
            ],
            storage_context=storage_context,
        )

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # similarity search
        search_embedding = OpenAIEmbedding().get_text_embedding("hello")
        q = VectorStoreQuery(
            query_embedding=search_embedding,
            similarity_top_k=1,
        )

        result = vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1

        # Identify the document to delete
        ref_id_to_delete = result.nodes[0].ref_doc_id

        # Delete the document
        vector_store.delete(ref_doc_id=ref_id_to_delete)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # Ensure that no results are returned
        result = vector_store.query(q)
        assert len(result.nodes) == 0

    def test_search_with_filter(
        self, vector_store: CouchbaseSearchVectorStore, node_embeddings: List[TextNode]
    ) -> None:
        """Test end to end Couchbase vector search with filter."""
        # Add nodes to the couchbase vector
        vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # similarity search
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("cake"),
            similarity_top_k=1,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="genre", value="Thriller", operator="=="),
                    MetadataFilter(key="pages", value=10, operator=">"),
                ]
            ),
        )

        result = vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].metadata.get("genre") == "Thriller"
            and result.nodes[0].metadata.get("pages") == 20
        )

    def test_hybrid_search(
        self, vector_store: CouchbaseSearchVectorStore, node_embeddings: List[TextNode]
    ) -> None:
        """Test the hybrid search functionality."""
        # Add nodes to the couchbase vector
        vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)

        # similarity search
        hybrid_query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )

        hybrid_result = vector_store.query(
            hybrid_query,
            cb_search_options={
                "query": {"field": "metadata.genre", "match": "Thriller"}
            },
        )

        assert result.nodes[0].get_content(
            metadata_mode=MetadataMode.NONE
        ) == hybrid_result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        assert result.similarities[0] <= hybrid_result.similarities[0]

    def test_output_fields(
        self, vector_store: CouchbaseSearchVectorStore, node_embeddings: List[TextNode]
    ) -> None:
        """Test the output fields functionality."""
        # Add nodes to the couchbase vector
        vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        q = VectorStoreQuery(
            query_embedding=text_to_embedding("cake"),
            similarity_top_k=1,
            output_fields=["text", "metadata.genre"],
        )

        result = vector_store.query(q)

        assert result.nodes is not None and len(result.nodes) == 1
        assert result.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "cake"
        assert result.nodes[0].metadata.get("genre") == "Thriller"


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


class TestCouchbaseVectorStore(TestCouchbaseSearchVectorStore):
    @pytest.fixture()
    def vector_store(self, cluster: Cluster) -> CouchbaseVectorStore:
        yield CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
        )
        delete_documents(cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)

    def test_deprecation_warning(self, cluster: Cluster) -> None:
        """Test that a deprecation warning is raised when instantiating CouchbaseVectorStore."""
        with pytest.warns(DeprecationWarning) as warnings_raised:
            CouchbaseVectorStore(
                cluster=cluster,
                bucket_name=BUCKET_NAME,
                scope_name=SCOPE_NAME,
                collection_name=COLLECTION_NAME,
                index_name=INDEX_NAME,
            )

        assert len(warnings_raised) >= 1, "DeprecationWarning was not raised."
