"""Azure CosmosDB NoSQL vCore Vector store index.

An index that is built on top of an existing vector store.

"""
import logging
import os
import requests
from typing import Any, Optional, Dict, cast, List

from azure.cosmos import CosmosClient, PartitionKey
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQueryResult, VectorStoreQuery
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from openai.lib.azure import AzureOpenAI
from openai.resources import Embeddings
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class AzureCosmosDBNoSqlVectorSearch(BasePydanticVectorStore):
    """Azure CosmosDB NoSQL vCore Vector Store.

    To use, you should have both:
    -the ``azure-cosmos`` python package installed
    -from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch
    """

    # _cosmosdb_client: Any = PrivateAttr()
    # _database_name: Any = PrivateAttr()
    # _container_name: Any = PrivateAttr()
    # _embedding: Any = PrivateAttr()
    # _vector_embedding_policy: Any = PrivateAttr()
    # _indexing_policy: Any = PrivateAttr()
    # _cosmos_container_properties: Any = PrivateAttr()
    # _cosmos_database_properties: Any = PrivateAttr()
    # _create_container: Any = PrivateAttr()

    def __init__(
            self,
            cosmos_client: Optional[Any] = None,
            vector_embedding_policy: Optional[Dict[str, Any]] = None,
            indexing_policy: Optional[Dict[str, Any]] = None,
            cosmos_container_properties: Optional[Dict[str, Any]] = None,
            cosmos_database_properties: Optional[Dict[str, Any]] = None,
            database_name: str = "vectorSearchDB",
            container_name: str = "vectorSearchContainer",
            create_container: bool = True,

    ) -> None:
        """Initialize the vector store.

        Args:
            cosmos_client: Client used to connect to azure cosmosdb no sql account.
            database_name: Name of the database to be created.
            container_name: Name of the container to be created.
            embedding: Text embedding model to use.
            vector_embedding_policy: Vector Embedding Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
            cosmos_database_properties: Database Properties for the container.
        """
        super().__init__()

        if self._cosmosdb_client is not None:
            self._cosmosdb_client = cast(CosmosClient, cosmos_client)

        if create_container:
            if (
                    indexing_policy["vectorIndexes"] is None
                    or len(indexing_policy["vectorIndexes"]) == 0
            ):
                raise ValueError(
                    "vectorIndexes cannot be null or empty in the indexing_policy."
                )
            if (
                    vector_embedding_policy is None
                    or len(vector_embedding_policy["vectorEmbeddings"]) == 0
            ):
                raise ValueError(
                    "vectorEmbeddings cannot be null "
                    "or empty in the vector_embedding_policy."
                )
            if (
                    self._cosmos_container_properties["partition_key"] is None
            ):
                raise ValueError(
                    "partition_key cannot be null "
                    "or empty for a container."
                )

        self._database_name = database_name
        self._container_name = container_name
        self._vector_embedding_policy = vector_embedding_policy
        self._indexing_policy = indexing_policy
        self._cosmos_container_properties = cosmos_container_properties
        self._cosmos_database_properties = cosmos_database_properties

        self._database = self._cosmos_client.create_database_if_not_exists(
            id=self._database_name,
            offer_throughput=self._cosmos_database_properties.get("offer_throughput"),
            session_token=self._cosmos_database_properties.get("session_token"),
            initial_headers=self._cosmos_database_properties.get("initial_headers"),
            etag=self._cosmos_database_properties.get("etag"),
            match_condition=self._cosmos_database_properties.get("match_condition"),
        )

        # Create the collection if it already doesn't exist
        self._container = self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=self._cosmos_container_properties["partition_key"],
            indexing_policy=self._indexing_policy,
            default_ttl=self._cosmos_container_properties.get("default_ttl"),
            offer_throughput=self._cosmos_container_properties.get("offer_throughput"),
            unique_key_policy=self._cosmos_container_properties.get("unique_key_policy"),
            conflict_resolution_policy=self._cosmos_container_properties.get("conflict_resolution_policy"),
            analytical_storage_ttl=self._cosmos_container_properties.get("analytical_storage_ttl"),
            computed_properties=self._cosmos_container_properties.get("computed_properties"),
            etag=self._cosmos_container_properties.get("etag"),
            match_condition=self._cosmos_container_properties.get("match_condition"),
            session_token=self._cosmos_container_properties.get("session_token"),
            initial_headers=self._cosmos_container_properties.get("initial_headers"),
            vector_embedding_policy=self._vector_embedding_policy,
        )

        self._embedding_key = self._vector_embedding_policy["vectorEmbeddings"][0]["path"][1:]

    def add(
            self,
            nodes: List[BaseNode],
            **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            A List of ids for successfully added nodes.

        """
        ids = []
        data_to_insert = []

        if not nodes:
            raise Exception("Texts can not be null or empty")

        #gets metadata from nodes
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )

            entry = {
                self._id_key: node.node_id,
                self._embedding_key: node.get_embedding(),
                self._text_key: node.get_content(metadata_mode=MetadataMode.NONE) or "",
                self._metadata_key: metadata,
            }
            data_to_insert.append(entry)
            ids.append(node.node_id)

        #inserts nodes into cosmos db
        for item in data_to_insert:
            self._container.create_item(item)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # delete by filtering on the doc_id metadata
        if ref_doc_id is None:
            raise ValueError("No id provided to delete.")

        self._container.delete_item(ref_doc_id)

    @property
    def client(self) -> Any:
        """Return CosmosDB client."""
        return self._cosmosdb_client

    def _query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        params: Dict[str, Any] = {
            "vector": query.query_embedding,
            "path": self._embedding_key,
            "k": query.similarity_top_k,
        }

        #I'm supposed to have an error here if the input is missing something

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for item in self._container.query_items(
                query='SELECT TOP @k c.id, VectorDistance(c.embedding,@embedding) AS SimilarityScore FROM c ORDER BY VectorDistance(c.embedding,@embedding)',
                parameters=[{"name": "@k", "value": VectorStoreQuery["similarity_top_k"]},
                            {"name": "@embedding", "value": VectorStoreQuery["query_embedding"]}],
                enable_cross_partition_query=True
        ):
            node = item
            node_id = item['id']
            node_score = item['SimilarityScore']

            top_k_ids.append(node_id)
            top_k_nodes.append(node)
            top_k_scores.append(node_score)

        result = VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

        return result

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query: a VectorStoreQuery object.

        Returns:
            A VectorStoreQueryResult containing the results of the query.
        """
        return self._query(query)


def main():
    # Creating a client based on azure portal details
    global cosmos_database_properties_test
    URL = ''
    KEY = ''
    client = CosmosClient(URL, credential=KEY)
    print(client)

    # Create the directory structure for the data if it doesn't exist
    os.makedirs('data/10k/', exist_ok=True)

    #Copilot:
    # Define the URL of the file to download
    file_url = 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf'

    # Define the local path where the file will be saved
    #local_file_path = 'data/10k/uber_2021.pdf'

    # Download the file and save it to the local path
    # response = requests.get(file_url)
    # if response.status_code == 200:
    #     with open(local_file_path, 'wb') as file:
    #         file.write(response.content)
    #     print(f"File downloaded and saved as {local_file_path}")
    # else:
    #     print(f"Failed to download the file. Status code: {response.status_code}")

    #llama index example:
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

    OPENAI_API_KEY = ""
    OPENAI_API_BASE = ""

    var = os.environ["OPENAI_API_KEY"] = ""

    openai_client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=OPENAI_API_BASE
    )

    embedding_test = Embeddings(openai_client)

    store = AzureCosmosDBNoSqlVectorSearch(cosmos_client=client,
                                           vector_embedding_policy=vector_embedding_policy,
                                           indexing_policy=indexing_policy,
                                           cosmos_container_properties=cosmos_container_properties_test,
                                           cosmos_database_properties=cosmos_database_properties_test,
                                           create_container=True,)


main()
