"""Test Couchbase Vector Search functionality."""

from __future__ import annotations
import os
from typing import Any, List

import pytest
import time

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


def get_cluster() -> Any:
    """Get a couchbase cluster object."""
    from datetime import timedelta

    from couchbase.auth import PasswordAuthenticator
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions

    auth = PasswordAuthenticator(USERNAME, PASSWORD)
    options = ClusterOptions(auth)
    connect_string = CONNECTION_STRING
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@pytest.fixture()
def cluster() -> Any:
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
            text="baz",
            id_="469e9537-7bc5-4669-9ff6-baa0ed086236",
            metadata={
                "genre": "Thriller",
                "pages": 20,
            },
            embedding=text_to_embedding("baz"),
        ),
    ]


@pytest.mark.skipif(
    not set_all_env_vars(), reason="missing Couchbase environment variables"
)
class TestCouchbaseSearchVectorStore:
    @classmethod
    def setup_method(self) -> None:
        self.cluster = get_cluster()
        # Delete all the documents in the collection
        delete_documents(self.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
        self.vector_store = CouchbaseSearchVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
        )

    def test_add_documents(self, node_embeddings: List[TextNode]) -> None:
        """Test adding documents to Couchbase vector store."""
        input_doc_ids = [node_embedding.id_ for node_embedding in node_embeddings]
        # Add nodes to the couchbase vector
        doc_ids = self.vector_store.add(node_embeddings)

        # Ensure that all nodes are returned & they are the same as input
        assert len(doc_ids) == len(node_embeddings)
        for doc_id in doc_ids:
            assert doc_id in input_doc_ids

    def test_search(self, node_embeddings: List[TextNode]) -> None:
        """Test end to end Couchbase vector search."""
        # Add nodes to the couchbase vector
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # similarity search
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

    def test_delete_doc(self) -> None:
        """Test delete document from Couchbase vector store."""
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Add nodes to the couchbase vector
        VectorStoreIndex.from_documents(
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

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1

        # Identify the document to delete
        ref_id_to_delete = result.nodes[0].ref_doc_id

        # Delete the document
        self.vector_store.delete(ref_doc_id=ref_id_to_delete)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # Ensure that no results are returned
        result = self.vector_store.query(q)
        assert len(result.nodes) == 0

    def test_search_with_filter(self, node_embeddings: List[TextNode]) -> None:
        """Test end to end Couchbase vector search with filter."""
        # Add nodes to the couchbase vector
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        # similarity search
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="genre", value="Thriller", operator="=="),
                    MetadataFilter(key="pages", value=10, operator=">"),
                ]
            ),
        )

        result = self.vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].metadata.get("genre") == "Thriller"
            and result.nodes[0].metadata.get("pages") == 20
        )

    def test_hybrid_search(self, node_embeddings: List[TextNode]) -> None:
        """Test the hybrid search functionality."""
        # Add nodes to the couchbase vector
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = self.vector_store.query(query)

        # similarity search
        hybrid_query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )

        hybrid_result = self.vector_store.query(
            hybrid_query,
            cb_search_options={
                "query": {"field": "metadata.genre", "match": "Thriller"}
            },
        )

        assert result.nodes[0].get_content(
            metadata_mode=MetadataMode.NONE
        ) == hybrid_result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        assert result.similarities[0] <= hybrid_result.similarities[0]

    def test_output_fields(self, node_embeddings: List[TextNode]) -> None:
        """Test the output fields functionality."""
        # Add nodes to the couchbase vector
        self.vector_store.add(node_embeddings)

        # Wait for the documents to be indexed
        time.sleep(SLEEP_DURATION)

        q = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
            output_fields=["text", "metadata.genre"],
        )

        result = self.vector_store.query(q)

        assert result.nodes is not None and len(result.nodes) == 1
        assert result.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "baz"
        assert result.nodes[0].metadata.get("genre") == "Thriller"


class TestCouchbaseVectorStore(TestCouchbaseSearchVectorStore):
    @classmethod
    def setup_method(self) -> None:
        self.cluster = get_cluster()
        # Delete all the documents in the collection
        delete_documents(self.cluster, BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)

        # Now, actually instantiate and assign to self.vector_store so inherited tests use it.
        # The warning has already been checked.
        self.vector_store = CouchbaseVectorStore(
            cluster=self.cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
        )

    def test_deprecation_warning(self) -> None:
        """Test that a deprecation warning is raised when instantiating CouchbaseVectorStore."""
        with pytest.warns(DeprecationWarning) as warnings_raised:
            CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=BUCKET_NAME,
                scope_name=SCOPE_NAME,
                collection_name=COLLECTION_NAME,
                index_name=INDEX_NAME,
            )

        assert len(warnings_raised) >= 1, "DeprecationWarning was not raised."
