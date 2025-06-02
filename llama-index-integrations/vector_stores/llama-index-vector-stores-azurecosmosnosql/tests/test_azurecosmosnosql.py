"""Test Azure CosmosDB NoSql Vector Search functionality."""

from __future__ import annotations

from time import sleep
from typing import List

import pytest

try:
    from azure.cosmos import CosmosClient, PartitionKey

    URL = "AZURE_COSMOSDB_URI"
    KEY = "AZURE_COSMOSDB_KEY"
    database_name = "test_database"
    container_name = "test_container"
    test_client = CosmosClient(URL, credential=KEY)

    indexing_policy = {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    }

    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 1536,
            }
        ]
    }

    partition_key = PartitionKey(path="/id")
    cosmos_container_properties_test = {"partition_key": partition_key}
    cosmos_database_properties_test = {}

    test_database = test_client.create_database_if_not_exists(id=database_name)
    test_container = test_database.create_container_if_not_exists(
        id=container_name,
        partition_key=partition_key,
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
    )

    cosmosnosql_available = True
except (ImportError, Exception):
    cosmosnosql_available = False

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch


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


@pytest.mark.skipif(not cosmosnosql_available, reason="cosmos client is not available")
class TestAzureCosmosNoSqlVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # insure the test container is empty
        items_list = test_container.read_all_items()
        first_item = next(iter(items_list), None)  # type: ignore[index]
        assert first_item is None

    @classmethod
    def teardown_class(cls) -> None:
        # delete all the items in the container
        for item in test_container.query_items(
            query="SELECT * FROM c", enable_cross_partition_query=True
        ):
            test_container.delete_item(item, partition_key=item["id"])

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the items in the container
        for item in test_container.query_items(
            query="SELECT * FROM c", enable_cross_partition_query=True
        ):
            test_container.delete_item(item, partition_key=item["id"])

    def test_add_and_delete(self) -> None:
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            database_name=database_name,
            container_name=container_name,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
        )
        sleep(1)  # waits for azure cosmos nosql to update
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

        items_amount = 0
        items_list = test_container.read_all_items()
        for item in items_list:
            items_amount += 1

        assert items_amount == 1

        vector_store.delete("test node id")

        items_amount = 0
        items_list = test_container.read_all_items()
        for item in items_list:
            items_amount += 1

        assert items_amount == 0

    def test_query(self, node_embeddings: List[TextNode]) -> None:
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)  # wait for azure cosmodb nosql to update the index

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
        print("res:\n", res)
        assert res.nodes
        assert res.nodes[0].get_content() == "lorem ipsum"

    def test_cosmos_client_with_host_and_key(
        self, node_embeddings: List[TextNode]
    ) -> None:
        vector_store = AzureCosmosDBNoSqlVectorSearch.from_host_and_key(
            host=URL,
            key=KEY,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)  # wait for azure cosmodb nosql to update the index

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
        print("res:\n", res)
        assert res.nodes
        assert res.nodes[0].get_content() == "lorem ipsum"

    def test_cosmos_client_with_connection_string(
        self, node_embeddings: List[TextNode]
    ) -> None:
        connection_string = "ACCOUNT_ENDPOINT=" + URL + ";" + "ACCOUNT_KEY=" + KEY + ";"
        vector_store = AzureCosmosDBNoSqlVectorSearch.from_connection_string(
            connection_string=connection_string,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
        )
