"""Test Azure DocumentDB Vector Search functionality."""

from __future__ import annotations

import os
from time import sleep
from typing import List
from uuid import uuid4

import pytest

try:
    from pymongo import MongoClient

    INDEX_NAME = "llamaindex-test-index"
    DB_NAME = "llamaindex_test_db"
    COLLECTION_NAME = f"llamaindex_test_collection_{uuid4().hex}"
    CONNECTION_STRING = os.environ.get("AZURE_DOCUMENTDB_URI")
    if CONNECTION_STRING is None:
        raise RuntimeError("AZURE_DOCUMENTDB_URI is not configured")
    test_client = MongoClient(CONNECTION_STRING)  # type: ignore
    collection = test_client[DB_NAME][COLLECTION_NAME]

    e2e_available = True
except (ImportError, Exception):
    e2e_available = False

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.azuredocumentdb import AzureDocumentDBVectorSearch


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


@pytest.mark.skipif(
    not e2e_available,
    reason="pymongo or AZURE_DOCUMENTDB_URI is not available",
)
class TestAzureDocumentDBVectorSearch:
    @classmethod
    def teardown_class(cls) -> None:
        collection.drop()  # type: ignore[index]
        test_client.close()  # type: ignore[possibly-undefined]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    def test_add_and_delete(self) -> None:
        vector_store = AzureDocumentDBVectorSearch(
            mongodb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
            cosmos_search_kwargs={"dimensions": 3},
        )
        sleep(1)  # waits for Azure DocumentDB to update
        vector_store.add(
            [
                TextNode(
                    text="test node text",
                    id_="test node id",
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc id")
                    },
                    embedding=[0.5, 0.5, 0.5],
                )
            ]
        )

        assert collection.count_documents({}) == 1

        vector_store.delete("test doc id")

        assert collection.count_documents({}) == 0

    def test_query(self, node_embeddings: List[TextNode]) -> None:
        vector_store = AzureDocumentDBVectorSearch(
            mongodb_client=test_client,  # type: ignore
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
            cosmos_search_kwargs={"dimensions": 3},
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)  # wait for Azure DocumentDB to update the index

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
        print("res:\n", res)
        sleep(5)
        assert res.nodes
        assert res.nodes[0].get_content() == "lorem ipsum"
