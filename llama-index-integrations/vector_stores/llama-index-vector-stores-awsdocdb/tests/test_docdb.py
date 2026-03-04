"""Test AWS DocumentDB Vector Store functionality."""

from __future__ import annotations

import os
from time import sleep

import pytest
from pymongo import MongoClient

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.awsdocdb import AWSDocDbVectorStore

INDEX_NAME = "llamaindex-test-index"
NAMESPACE = "llamaindex_test_db.llamaindex_test_collection"
CONNECTION_STRING = os.getenv("DOCUMENTDB_URI", "")
DB_NAME, COLLECTION_NAME = "mydatabase", "customers"

pymongo_available = True

if CONNECTION_STRING == "":
    pymongo_available = False


"""
Test instructions:
Add a DocumentDB connection string to your environment after creating an instance
Follow the instructions at https://docs.aws.amazon.com/documentdb/latest/developerguide/connect_programmatically.html to connect to your instance
Use the contributor steps at https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md to setup the environment and run all the tests here
"""


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


try:
    test_client = MongoClient(CONNECTION_STRING)  # type: ignore
    collection = test_client[DB_NAME][COLLECTION_NAME]
except Exception:
    test_client = None
    collection = None


@pytest.mark.skipif(
    test_client is None, reason="A DocumentDB instance has not been configured"
)
class TestAWSDocDBVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # insure the test collection is empty
        assert collection.count_documents({}) == 0  # type: ignore[index]

    @classmethod
    def teardown_class(cls) -> None:
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    def test_add_and_delete(self) -> None:
        vector_store = AWSDocDbVectorStore(
            docdb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
        )
        sleep(1)  # waits for mongot to update Lucene's index
        x = vector_store.add(
            [
                TextNode(
                    text="test node text",
                    id_="test node id",
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc id")
                    },
                    embedding=[0.5, 0.5],
                )
            ]
        )

        assert collection.count_documents({}) == 1
        vector_store.delete(x[0])
        assert collection.count_documents({}) == 0

    def test_query_default(self, node_embeddings: list[TextNode]) -> None:
        # tests cosine similarity
        vector_store = AWSDocDbVectorStore(
            docdb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)  # wait for mongot to update the index

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )

        assert res.nodes
        assert res.nodes[0].get_content() == "lorem ipsum"

    def test_query_dot(self, node_embeddings: list[TextNode]) -> None:
        vector_store = AWSDocDbVectorStore(
            docdb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            similarity_score="dotProduct",
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)  # wait for mongot to update the index

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
        assert res.nodes
        assert res.nodes[0].get_content() == "lorem ipsum"

    def test_query_euclidean(self, node_embeddings: list[TextNode]) -> None:
        vector_store = AWSDocDbVectorStore(
            docdb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            similarity_score="euclidean",
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)  # wait for mongot to update the index

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
        assert res.nodes
        assert res.nodes[0].get_content() == "lorem ipsum"

    def test_embeddings(self):
        pass

    def test_index_cd(self):
        vector_store = AWSDocDbVectorStore(
            docdb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
        )
        vector_store.create_index(3)
        vector_store.delete_index()
