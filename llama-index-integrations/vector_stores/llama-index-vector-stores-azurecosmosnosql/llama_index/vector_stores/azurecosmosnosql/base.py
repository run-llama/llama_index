"""
Azure CosmosDB NoSQL vCore Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, Optional, Dict, cast, List

from azure.identity import ClientSecretCredential
from azure.cosmos import CosmosClient
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)
USER_AGENT = ("LlamaIndex-CDBNoSql-VectorStore-Python",)


class AzureCosmosDBNoSqlVectorSearch(BasePydanticVectorStore):
    """
    Azure CosmosDB NoSQL vCore Vector Store.

    To use, you should have both:
    -the ``azure-cosmos`` python package installed
    -from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch
    """

    stores_text: bool = True
    flat_metadata: bool = True

    _cosmos_client: Any = PrivateAttr()
    _database_name: Any = PrivateAttr()
    _container_name: Any = PrivateAttr()
    _embedding_key: Any = PrivateAttr()
    _vector_embedding_policy: Any = PrivateAttr()
    _indexing_policy: Any = PrivateAttr()
    _cosmos_container_properties: Any = PrivateAttr()
    _cosmos_database_properties: Any = PrivateAttr()
    _create_container: Any = PrivateAttr()
    _database: Any = PrivateAttr()
    _container: Any = PrivateAttr()
    _id_key: Any = PrivateAttr()
    _text_key: Any = PrivateAttr()
    _metadata_key: Any = PrivateAttr()

    def __init__(
        self,
        cosmos_client: CosmosClient,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Optional[Dict[str, Any]] = None,
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        create_container: bool = True,
        id_key: str = "id",
        text_key: str = "text",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the vector store.

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

        if cosmos_client is not None:
            self._cosmos_client = cast(CosmosClient, cosmos_client)

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
                cosmos_container_properties is None
                or cosmos_container_properties["partition_key"] is None
            ):
                raise ValueError(
                    "partition_key cannot be null or empty for a container."
                )

        self._database_name = database_name
        self._container_name = container_name
        self._vector_embedding_policy = vector_embedding_policy
        self._indexing_policy = indexing_policy
        self._cosmos_container_properties = cosmos_container_properties
        self._cosmos_database_properties = cosmos_database_properties
        self._id_key = id_key
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._embedding_key = self._vector_embedding_policy["vectorEmbeddings"][0][
            "path"
        ][1:]

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
            unique_key_policy=self._cosmos_container_properties.get(
                "unique_key_policy"
            ),
            conflict_resolution_policy=self._cosmos_container_properties.get(
                "conflict_resolution_policy"
            ),
            analytical_storage_ttl=self._cosmos_container_properties.get(
                "analytical_storage_ttl"
            ),
            computed_properties=self._cosmos_container_properties.get(
                "computed_properties"
            ),
            etag=self._cosmos_container_properties.get("etag"),
            match_condition=self._cosmos_container_properties.get("match_condition"),
            session_token=self._cosmos_container_properties.get("session_token"),
            initial_headers=self._cosmos_container_properties.get("initial_headers"),
            vector_embedding_policy=self._vector_embedding_policy,
        )

    @classmethod
    def from_host_and_key(
        cls,
        host: str,
        key: str,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Optional[Dict[str, Any]] = None,
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        create_container: bool = True,
        id_key: str = "id",
        text_key: str = "text",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> "AzureCosmosDBNoSqlVectorSearch":
        """Initialize the vector store using the cosmosDB host and key."""
        cosmos_client = CosmosClient(host, key, user_agent=USER_AGENT)
        return cls(
            cosmos_client,
            vector_embedding_policy,
            indexing_policy,
            cosmos_container_properties,
            cosmos_database_properties,
            database_name,
            container_name,
            create_container,
            id_key,
            text_key,
            metadata_key,
            **kwargs,
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Optional[Dict[str, Any]] = None,
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        create_container: bool = True,
        id_key: str = "id",
        text_key: str = "text",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> "AzureCosmosDBNoSqlVectorSearch":
        """Initialize the vector store using the cosmosDB connection string."""
        cosmos_client = CosmosClient.from_connection_string(
            connection_string, user_agent=USER_AGENT
        )
        return cls(
            cosmos_client,
            vector_embedding_policy,
            indexing_policy,
            cosmos_container_properties,
            cosmos_database_properties,
            database_name,
            container_name,
            create_container,
            id_key,
            text_key,
            metadata_key,
            **kwargs,
        )

    @classmethod
    def from_uri_and_managed_identity(
        cls,
        cosmos_uri: str,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Optional[Dict[str, Any]] = None,
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        create_container: bool = True,
        id_key: str = "id",
        text_key: str = "text",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> "AzureCosmosDBNoSqlVectorSearch":
        """Initialize the vector store using the cosmosDB uri and managed identity."""
        cosmos_client = CosmosClient(
            cosmos_uri, ClientSecretCredential, user_agent=USER_AGENT
        )
        return cls(
            cosmos_client,
            vector_embedding_policy,
            indexing_policy,
            cosmos_container_properties,
            cosmos_database_properties,
            database_name,
            container_name,
            create_container,
            id_key,
            text_key,
            metadata_key,
            **kwargs,
        )

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            A List of ids for successfully added nodes.

        """
        ids = []
        data_to_insert = []

        if not nodes:
            raise Exception("Texts can not be null or empty")

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

        for item in data_to_insert:
            self._container.upsert_item(item)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        items = self._container.query_items(
            query=f"SELECT c.id, c.id AS partitionKey FROM c WHERE c.{self._metadata_key}.ref_doc_id = '{ref_doc_id}'",
            enable_cross_partition_query=True,
        )
        for item in items:
            self._container.delete_item(item["id"], partition_key=item["partitionKey"])

    @property
    def client(self) -> Any:
        """Return CosmosDB client."""
        return self._cosmos_client

    def _query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        params: Dict[str, Any] = {
            "vector": query.query_embedding,
            "path": self._embedding_key,
            "k": query.similarity_top_k,
        }

        pre_filter = kwargs.get("pre_filter", {})

        query = "SELECT "

        # If limit_offset_clause is not specified, add TOP clause
        if pre_filter is None or pre_filter.get("limit_offset_clause") is None:
            query += f"TOP {params.get('k', 2)} "

        query += (
            "c.id, c.text, c.metadata, "
            f"VectorDistance(c.{self._embedding_key}, @embeddings) AS SimilarityScore FROM c"
        )

        # Add where_clause if specified
        if pre_filter is not None and pre_filter.get("where_clause") is not None:
            query += " {}".format(pre_filter["where_clause"])

        query += f" ORDER BY VectorDistance(c.{self._embedding_key}, @embeddings)"

        # Add limit_offset_clause if specified
        if pre_filter is not None and pre_filter.get("limit_offset_clause") is not None:
            query += " {}".format(pre_filter["limit_offset_clause"])
        parameters = [
            {"name": "@embeddings", "value": params["vector"]},
        ]

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for item in self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        ):
            node = metadata_dict_to_node(item[self._metadata_key])
            node.set_content(item[self._text_key])

            node_id = item[self._id_key]
            node_score = item["SimilarityScore"]

            top_k_ids.append(node_id)
            top_k_nodes.append(node)
            top_k_scores.append(node_score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: a VectorStoreQuery object.

        Returns:
            A VectorStoreQueryResult containing the results of the query.

        """
        return self._query(query, **kwargs)
