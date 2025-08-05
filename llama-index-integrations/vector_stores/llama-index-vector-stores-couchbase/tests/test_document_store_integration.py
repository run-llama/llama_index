# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from datetime import datetime
from typing import List

from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.dataclasses import Document, ByteStream
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from couchbase_haystack import (
    CouchbaseQueryDocumentStore,
    QueryVectorSearchType,
    QueryVectorSearchFunctionParams,
)
from couchbase_haystack.document_stores.auth import CouchbasePasswordAuthenticator
from couchbase_haystack.document_stores.cluster_options import CouchbaseClusterOptions
from couchbase.options import KnownConfigProfiles
from couchbase.exceptions import (
    ScopeAlreadyExistsException,
    CollectionAlreadyExistsException,
)
from couchbase.options import QueryOptions
from datetime import timedelta
from sentence_transformers import SentenceTransformer
import time
import json
from pandas import DataFrame
from uuid import uuid1

model = SentenceTransformer("all-MiniLM-L6-v2")

# Test configuration
TEST_BUCKET = "test_bucket"
TEST_SCOPE = "test_scope"
TEST_COLLECTION = "test_collection"
TEST_INDEX = "test_vector_index"
VECTOR_DIMENSION = 384


class TestGSIDocumentStoreIntegration(DocumentStoreBaseTests):
    @pytest.fixture(scope="class")
    def sample_init_documents(self) -> List[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id=f"doc_init_{i}",
                content=f"Test document {i}",
                meta={
                    "field1": f"value{i}",
                    "field2": i,
                    "created_at": datetime.now().isoformat(),
                },
                embedding=[0.001 * i] * VECTOR_DIMENSION,
            )
            for i in range(2048)
        ]

    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id=f"doc_{i}",
                content=f"Test document {i}",
                meta={
                    "field1": f"value{i}",
                    "field2": i,
                    "created_at": datetime.now().isoformat(),
                },
                embedding=[0.001 * i] * VECTOR_DIMENSION,
            )
            for i in range(1024)
        ]

    @pytest.fixture(scope="class")
    def document_store_with_index_creation(self, sample_init_documents):
        # Create authenticator
        authenticator = CouchbasePasswordAuthenticator(
            username=Secret.from_env_var("USER_NAME"),
            password=Secret.from_env_var("PASSWORD"),
        )

        # Create cluster options
        cluster_options = CouchbaseClusterOptions(
            protocol=KnownConfigProfiles.WanDevelopment
        )

        # Create document store
        store = CouchbaseQueryDocumentStore(
            cluster_connection_string=Secret.from_env_var("CONNECTION_STRING"),
            authenticator=authenticator,
            cluster_options=cluster_options,
            bucket=TEST_BUCKET,
            scope=TEST_SCOPE,
            collection=TEST_COLLECTION,
            index_name=TEST_INDEX,
            query_vector_search_params=QueryVectorSearchFunctionParams(
                search_type=QueryVectorSearchType.ANN,
                dimension=VECTOR_DIMENSION,
                similarity="L2",
            ),
            vector_field="embedding",
        )

        # Create scope if it doesn't exist
        try:
            store.bucket.collections().create_scope(scope_name=TEST_SCOPE)
        except ScopeAlreadyExistsException:
            pass

        # Create collection if it doesn't exist
        try:
            store.bucket.collections().create_collection(
                collection_name=TEST_COLLECTION, scope_name=TEST_SCOPE
            )
        except CollectionAlreadyExistsException:
            pass

        # Write initial documents
        store.write_documents(sample_init_documents, policy=DuplicatePolicy.OVERWRITE)

        with_opts = json.dumps(
            {
                "dimension": VECTOR_DIMENSION,
                "description": "IVF1024,PQ32x8",
                "similarity": "L2",
            }
        )
        # Create index before tests
        result = store.scope.query(
            f"""
                CREATE INDEX {TEST_INDEX}
                ON {TEST_BUCKET}.{TEST_SCOPE}.{TEST_COLLECTION} ({store.vector_field} VECTOR)
                USING GSI WITH {with_opts}
                """,
            QueryOptions(timeout=timedelta(seconds=300)),
        ).execute()
        print(result)
        # time.sleep(60)

        store.delete_documents([doc.id for doc in store.filter_documents()])

        yield store
        store.bucket.collections().drop_collection(
            collection_name=TEST_COLLECTION, scope_name=TEST_SCOPE
        )
        # Cleanup after tests
        store.bucket.close()

    @pytest.fixture()
    def document_store(self, document_store_with_index_creation):
        yield document_store_with_index_creation
        document_store_with_index_creation.delete_documents(
            [doc.id for doc in document_store_with_index_creation.filter_documents()]
        )

    def assert_documents_are_equal(
        self, received: List[Document], expected: List[Document]
    ):
        print(received, expected)
        for r in received:
            r.score = None
            r.embedding = None
        received_dict = {doc.id: doc for doc in received}
        received = []
        for doc in expected:
            received.append(received_dict.get(doc.id))
            doc.embedding = None
        print("================")
        print(received, expected)
        print(len(received), len(expected))
        # print([doc.to_dict(flatten=False) if doc else doc for doc in received])
        # print([doc.to_dict(flatten=False) for doc in expected])
        super().assert_documents_are_equal(received, expected)

    def test_write_documents_duplicate_skip(self, document_store):
        pass

    def test_no_filters(self, document_store: CouchbaseQueryDocumentStore):
        """Test filter_documents() with empty filters"""
        self.assert_documents_are_equal(document_store.filter_documents(), [])
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), [])
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        self.assert_documents_are_equal(document_store.filter_documents(), docs)
        self.assert_documents_are_equal(
            document_store.filter_documents(filters={}), docs
        )

    def test_write_documents(self, document_store: CouchbaseQueryDocumentStore):
        documents = [
            Document(id=uuid1().hex, content="Haystack is an amazing tool for search."),
            Document(
                id=uuid1().hex,
                content="We are using pre-trained models to generate embeddings.",
            ),
            Document(id=uuid1().hex, content="The weather is sunny today."),
        ]
        for doc in documents:
            embedding = model.encode(doc.content).tolist()
            doc.embedding = embedding

        assert document_store.write_documents(documents) == 3
        retrieved_docs = document_store.filter_documents()
        assert len(retrieved_docs) == 3
        retrieved_docs.sort(key=lambda x: x.id)
        self.assert_documents_are_equal(retrieved_docs, documents)

    def test_write_blob(self, document_store: CouchbaseQueryDocumentStore):
        bytestream = ByteStream(
            b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type"
        )
        documents = [Document(blob=bytestream)]
        for doc in documents:
            # Assuming blob_content is in bytes, decode it to string if necessary
            embedding = model.encode(bytestream.data.decode("utf-8")).tolist()
            doc.embedding = embedding
        assert document_store.write_documents(documents) == 1
        retrieved_docs = document_store.filter_documents()
        time.sleep(30)
        self.assert_documents_are_equal(retrieved_docs, documents)

    def test_write_dataframe(self, document_store: CouchbaseQueryDocumentStore):
        dataframe = DataFrame({"col1": [1, 2], "col2": [3, 4]})
        docs = [Document(dataframe=dataframe)]
        document_store.write_documents(docs)
        retrieved_docs = document_store.filter_documents()
        self.assert_documents_are_equal(retrieved_docs, docs)

    def test_comparison_in1(
        self, document_store: CouchbaseQueryDocumentStore, filterable_docs
    ):
        """Test filter_documents() with 'in' comparator"""
        document_store.write_documents(filterable_docs)
        # time.sleep(2000)
        result = document_store.filter_documents(
            {"field": "meta.number", "operator": "in", "value": [10, -10]}
        )
        assert len(result)
        expected = [
            d
            for d in filterable_docs
            if d.meta.get("number") is not None and d.meta["number"] in [10, -10]
        ]
        self.assert_documents_are_equal(result, expected)

    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.page", "operator": "==", "value": "90"},
                        {
                            "field": "meta.chapter",
                            "operator": "==",
                            "value": "conclusion",
                        },
                    ],
                },
            ],
        }

        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.meta.get("number") == 100 and d.meta.get("chapter") == "intro")
                or (
                    d.meta.get("page") == "90" and d.meta.get("chapter") == "conclusion"
                )
            ],
        )

    def test_duplicate_document_handling(self, document_store, sample_documents):
        """Test handling of duplicate documents."""
        # Write documents first time
        document_store.write_documents(sample_documents)

        # Try to write same documents again with FAIL policy
        with pytest.raises(Exception):
            document_store.write_documents(
                sample_documents, policy=DuplicatePolicy.FAIL
            )

        # Write with OVERWRITE policy
        document_store.write_documents(
            sample_documents, policy=DuplicatePolicy.OVERWRITE
        )

        # Verify document count hasn't changed
        documents = document_store.filter_documents()
        assert len(documents) == len(sample_documents)

    def test_vector_search(
        self, document_store: CouchbaseQueryDocumentStore, sample_documents
    ):
        """Test vector search functionality."""
        # Write documents
        document_store.write_documents(sample_documents)

        # Create a query embedding
        query_embedding = [0.1] * VECTOR_DIMENSION

        # Perform vector search
        results = document_store.vector_search(query_embedding, top_k=3)

        # Verify results
        assert len(results) == 3
        assert all(hasattr(doc, "score") for doc in results)
        print(results)
        assert all(doc.score is not None for doc in results)

        # TODO: ADD logic to check if the results are correct

    def test_vector_search_with_filters(self, document_store, sample_documents):
        """Test vector search with filters."""
        # Write documents
        document_store.write_documents(sample_documents)

        # Create a query embedding
        query_embedding = [0.1] * VECTOR_DIMENSION

        # Define filters
        filters = {"field": "field2", "operator": ">", "value": 2}

        # Perform vector search with filters
        results = document_store.vector_search(
            query_embedding, top_k=3, filters=filters
        )

        # Verify results
        assert len(results) <= 3
        assert all(doc.meta["field2"] > 2 for doc in results)
