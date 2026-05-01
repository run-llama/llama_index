"""
Azure CosmosDB NoSQL vCore Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import re
from typing import Any, Optional, Dict, cast, List, Tuple

from azure.identity import ClientSecretCredential
from azure.cosmos import CosmosClient
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.azurecosmosnosql.utils import (
    Constants,
    ParamMapping,
    AzureCosmosDBNoSqlVectorSearchType,
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
    _table_alias: Any = PrivateAttr()
    _full_text_search_enabled: Any = PrivateAttr()

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
        table_alias: str = "c",
        id_key: str = "id",
        text_key: str = "text",
        metadata_key: str = "metadata",
        full_text_search_enabled: bool = False,
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
            create_container: Set to true if the container does not exist.
            table_alias: Alias for the table to be created.
            id_key: The key to use for the id field in the container.
            text_key: The key to use for the text field in the container.
            metadata_key: The key to use for the metadata field in the container.
            full_text_search_enabled: Set to true if the full text search should be enabled.

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
        self._table_alias = table_alias
        self._id_key = id_key
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._embedding_key = self._vector_embedding_policy["vectorEmbeddings"][0][
            "path"
        ][1:]
        # Distance function from container vector policy. Determines threshold
        # comparison direction: 'euclidean' is a distance (lower = more similar),
        # 'cosine' / 'dotproduct' are similarities (higher = more similar).
        # Default per Azure Cosmos DB NoSQL docs: cosine.
        # See: https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
        self._distance_function = (
            self._vector_embedding_policy["vectorEmbeddings"][0]
            .get("distanceFunction", "cosine")
            .lower()
        )
        self._full_text_search_enabled = full_text_search_enabled

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
            full_text_policy=self._cosmos_container_properties.get("full_text_policy"),
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
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            database_name=database_name,
            container_name=container_name,
            create_container=create_container,
            id_key=id_key,
            text_key=text_key,
            metadata_key=metadata_key,
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
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            database_name=database_name,
            container_name=container_name,
            create_container=create_container,
            id_key=id_key,
            text_key=text_key,
            metadata_key=metadata_key,
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
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            database_name=database_name,
            container_name=container_name,
            create_container=create_container,
            id_key=id_key,
            text_key=text_key,
            metadata_key=metadata_key,
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

    def _is_hybrid_search_type(self, search_type: str) -> bool:
        """Check if the search type is a hybrid (RRF) search type."""
        return search_type == AzureCosmosDBNoSqlVectorSearchType.HYBRID

    def _is_full_text_search_type(self, search_type: str) -> bool:
        """Check if the search type is a full text search type."""
        return search_type in (
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_SEARCH,
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
        )

    def _is_vector_search_type(self, search_type: str) -> bool:
        """Check if the search type requires vector search with vector embeddings."""
        return search_type in (
            AzureCosmosDBNoSqlVectorSearchType.VECTOR,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
        )

    def _validate_search_args(
        self,
        search_type: str,
        vector: Optional[List[float]] = None,
        return_with_vectors: bool = False,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        offset_limit: Optional[str] = None,
        weights: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Validate search arguments."""
        # Validate search_type using CosmosDBVectorStoreSearchType enum
        try:
            AzureCosmosDBNoSqlVectorSearchType(search_type)
        except ValueError:
            valid_options = ", ".join(
                t.value for t in AzureCosmosDBNoSqlVectorSearchType
            )
            raise ValueError(
                f"Invalid search_type '{search_type}'. "
                f"Valid options are: {valid_options}."
            )

        if self._full_text_search_enabled is False and self._is_full_text_search_type(
            search_type
        ):
            raise ValueError(
                f"Full text search is not enabled for this collection, "
                f"cannot perform search_type '{search_type}'."
            )

        # Validate vector and return_with_vectors
        if self._is_vector_search_type(search_type):
            if vector is None:
                raise ValueError(
                    f"Embedding must be provided for search_type '{search_type}'."
                )
        else:
            if return_with_vectors:
                raise ValueError(
                    "'return_with_vectors' can only be True for vector search types using vector embeddings."
                )

        # full_text_rank_filter is required for full-text-ranking and hybrid
        if (
            search_type == AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING
            or self._is_hybrid_search_type(search_type)
        ) and not full_text_rank_filter:
            raise ValueError(
                f"'full_text_rank_filter' is required for search_type '{search_type}'."
            )

        # Validate offset_limit format if provided
        if offset_limit and not re.match(
            r"^\s*OFFSET\s+\d+\s+LIMIT\s+\d+\s*$", offset_limit, re.IGNORECASE
        ):
            raise ValueError(
                f"Invalid 'offset_limit' format: {offset_limit!r}. "
                "Expected: 'OFFSET <int> LIMIT <int>'."
            )

        # Validate weights: only for hybrid (server-side weighted RRF), and
        # length must match number of RRF components (one per
        # full_text_rank_filter entry + one for the vector component).
        if weights is not None:
            if not self._is_hybrid_search_type(search_type):
                raise ValueError("'weights' is only supported for 'hybrid' search.")
            expected = (len(full_text_rank_filter) if full_text_rank_filter else 0) + 1
            if len(weights) != expected:
                raise ValueError(
                    f"'weights' must have {expected} elements "
                    f"(one per full_text_rank_filter entry + 1 for the vector "
                    f"component); got {len(weights)}."
                )

        # Validate threshold: only meaningful for vector-bearing search types.
        if threshold is not None and not self._is_vector_search_type(search_type):
            raise ValueError(
                "'threshold' is only supported for search types that produce a "
                "per-row similarity score ('vector' or 'hybrid')."
            )

    def _generate_projection_fields(
        self,
        search_type: str,
        param_mapping: ParamMapping,
        projection_mapping: Optional[Dict[str, Any]] = None,
        return_with_vectors: bool = False,
        vector: Optional[List[float]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        is_hybrid = self._is_hybrid_search_type(search_type)
        if projection_mapping:
            projection_fields = [
                f"{self._table_alias}.{key} as {alias}"
                for key, alias in projection_mapping.items()
            ]
        elif (
            search_type == AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING
            or is_hybrid
        ):
            # Use direct field path syntax (c.fieldname) for the base field
            # projection in ORDER BY RANK queries. Bracket-indexer + parameterised
            # values are accepted by the engine here too, but direct paths keep
            # these queries closer to the patterns shown in the public docs.
            projection_fields = [
                f"{self._table_alias}.{Constants.ID} as {Constants.ID}"
            ]
            projection_fields += [
                f"{self._table_alias}.{key} as {key}"
                for key in [self._text_key, self._metadata_key]
            ]
            if (
                search_type == AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING
                and full_text_rank_filter
            ):
                seen = {self._text_key, self._metadata_key}
                for item in full_text_rank_filter:
                    field = item[Constants.SEARCH_FIELD]
                    if field not in seen:
                        projection_fields.append(
                            f"{self._table_alias}.{field} as {field}"
                        )
                        seen.add(field)
        else:
            projection_fields = [
                f"{self._table_alias}.{Constants.ID} as {Constants.ID}"
            ]
            projection_fields += [
                param_mapping.gen_proj_field(key=key, value=key, alias=key)
                for key in [self._text_key, self._metadata_key]
            ]

        # If it's a vector search type, include vector distance projection and optionally the vector itself
        if self._is_vector_search_type(search_type):
            if return_with_vectors:
                projection_fields.append(
                    param_mapping.gen_proj_field(
                        key=Constants.VECTOR_KEY,
                        value=self._embedding_key,
                        alias=AzureCosmosDBNoSqlVectorSearchType.VECTOR.value,
                    )
                )
            # Always project SimilarityScore for vector-bearing search types.
            # CosmosDB accepts bracket-indexer + parameterised vectors in the
            # SELECT projection of every query, including those that use
            # ORDER BY RANK RRF. Surfacing the per-row vector distance is
            # essentially free (the value is already computed during the
            # search) and is useful both for caller-side relevance display
            # and for the optional ``threshold`` post-filter applied in
            # ``_execute_search_query``. The only ORDER BY RANK restriction
            # worth noting is on FullTextScore string literals; see
            # _generate_order_by_component_with_full_text_rank_filter.
            projection_fields.append(
                param_mapping.gen_vector_distance_proj_field(
                    vector_field=self._embedding_key,
                    vector=vector,
                    alias=Constants.SIMILARITY_SCORE,
                )
            )
        return f" {', '.join(projection_fields)}"

    def _generate_limit_clause(self, param_mapping: ParamMapping, limit: int) -> str:
        """Generate TOP limit clause for the query."""
        limit_key = param_mapping.gen_param_key(key=Constants.LIMIT, value=limit)
        return f" TOP {limit_key}"

    def _generate_order_by_component_with_full_text_rank_filter(
        self,
        full_text_rank_filter: Dict[str, str],
    ) -> str:
        """Generate ORDER BY component for full text rank filter."""
        search_field = full_text_rank_filter[Constants.SEARCH_FIELD]
        search_text = full_text_rank_filter[Constants.SEARCH_TEXT]
        # Use direct field path (self._table_alias.fieldname) —
        # bracket indexer is not supported in CosmosDB full text ORDER BY RANK.
        search_proj_field = f"{self._table_alias}.{search_field}"
        # CosmosDB requires string literals in FullTextScore inside ORDER BY RANK —
        # parameterized values are rejected with error code 2206.
        terms = [f"'{term}'" for term in search_text.split()]
        return f"FullTextScore({search_proj_field}, {', '.join(terms)})"

    def _generate_order_by_clause(
        self,
        search_type: str,
        param_mapping: ParamMapping,
        vector: Optional[List[float]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        weights: Optional[List[float]] = None,
    ) -> str:
        order_by_clause = ""
        if search_type == AzureCosmosDBNoSqlVectorSearchType.VECTOR:
            vector_distance_proj_field = param_mapping.gen_vector_distance_proj_field(
                vector_field=self._embedding_key, vector=vector
            )
            order_by_clause = f"ORDER BY {vector_distance_proj_field}"
        elif search_type in (
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
        ):
            # full_text_rank_filter is required here — already validated upstream
            # in _validate_search_args.
            components = [
                self._generate_order_by_component_with_full_text_rank_filter(
                    full_text_rank_filter=item,
                )
                for item in full_text_rank_filter or []
            ]
            if self._is_hybrid_search_type(search_type):
                components.append(
                    param_mapping.gen_vector_distance_order_by_field(
                        vector_field=self._embedding_key, vector=vector
                    )
                )
            if search_type == AzureCosmosDBNoSqlVectorSearchType.HYBRID and weights:
                # Server-side weighted RRF: weights are passed as the last
                # argument to RRF as an inline array literal.
                # See https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search
                weights_literal = "[" + ", ".join(str(w) for w in weights) + "]"
                rrf_args = ", ".join(components) + f", {weights_literal}"
                order_by_clause = f"ORDER BY RANK RRF({rrf_args})"
            elif len(components) == 1:
                order_by_clause = f"ORDER BY RANK {components[0]}"
            else:
                rrf_args = ", ".join(components)
                order_by_clause = f"ORDER BY RANK RRF({rrf_args})"
        return f" {order_by_clause}"

    def _construct_search_query(
        self,
        limit: int,
        search_type: str,
        vector: Optional[List[float]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        offset_limit: Optional[str] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        return_with_vectors: bool = False,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Construct the search query and parameters."""
        query = "SELECT"
        param_mapping = ParamMapping(table=self._table_alias)

        is_hybrid = self._is_hybrid_search_type(search_type)

        # ORDER BY RANK RRF does not support TOP — use OFFSET/LIMIT for hybrid types.
        if not offset_limit and not is_hybrid:
            query += self._generate_limit_clause(
                param_mapping=param_mapping, limit=limit
            )

        query += self._generate_projection_fields(
            search_type=search_type,
            param_mapping=param_mapping,
            projection_mapping=projection_mapping,
            return_with_vectors=return_with_vectors,
            vector=vector,
            full_text_rank_filter=full_text_rank_filter,
        )

        query += f" FROM {self._table_alias}"

        if where:
            query += f" WHERE {where}"

        if search_type != AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_SEARCH:
            order_by_clause = self._generate_order_by_clause(
                search_type=search_type,
                param_mapping=param_mapping,
                vector=vector,
                full_text_rank_filter=full_text_rank_filter,
                weights=weights,
            )
            query += order_by_clause

        if offset_limit:
            query += f" {offset_limit}"
        elif is_hybrid:
            query += f" OFFSET 0 LIMIT {limit}"

        parameters = param_mapping.export_parameter_list()
        return query, parameters

    def _execute_search_query(
        self,
        query: str,
        search_type: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        return_with_vectors: bool = False,
        projection_mapping: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> VectorStoreQueryResult:
        """Execute the search query and return results."""
        parameters = parameters if parameters else []

        # Execute the query
        items = self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )

        # Filter items if a threshold was supplied for a vector-bearing search.
        # Threshold semantics depend on the container's distanceFunction:
        #   * cosine / dotproduct → similarity score (higher = more similar) → keep score > threshold
        #   * euclidean → distance (lower = more similar) → keep score < threshold
        # Strict comparison: an item with a score equal to the threshold is excluded.
        # Pass ``threshold=None`` (or omit it) to disable filtering.
        if (
            threshold is not None
            and search_type
            and self._is_vector_search_type(search_type)
        ):
            if self._distance_function == "euclidean":
                filtered_items = [
                    item
                    for item in items
                    if item.get(Constants.SIMILARITY_SCORE, 0.0) < threshold
                ]
            else:
                filtered_items = [
                    item
                    for item in items
                    if item.get(Constants.SIMILARITY_SCORE, 0.0) > threshold
                ]
        else:
            filtered_items = list(items)

        # Build VectorStoreQueryResult
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for item in filtered_items:
            node_id = item.get(self._id_key, "")
            node_score = item.get(Constants.SIMILARITY_SCORE, 0.0)

            if projection_mapping:
                # Projected items don't have the standard metadata structure;
                # return a TextNode with the projected fields surfaced as metadata.
                node = TextNode(
                    id_=node_id,
                    metadata={
                        alias: item.get(alias) for alias in projection_mapping.values()
                    },
                )
            else:
                raw_metadata = item.get(self._metadata_key, {})
                if raw_metadata and raw_metadata.get("_node_content"):
                    node = metadata_dict_to_node(raw_metadata)
                    node.set_content(item.get(self._text_key, ""))
                else:
                    # full_text_ranking and similar queries only project specific fields
                    # (e.g. id + text) without the full metadata structure — build a plain TextNode
                    node = TextNode(
                        id_=node_id,
                        text=item.get(self._text_key, ""),
                        metadata=raw_metadata if isinstance(raw_metadata, dict) else {},
                    )

            top_k_ids.append(node_id)
            top_k_nodes.append(node)
            top_k_scores.append(node_score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def _search_query(
        self,
        vectors: Optional[List[float]] = None,
        limit: int = 5,
        search_type: str = AzureCosmosDBNoSqlVectorSearchType.VECTOR,
        return_with_vectors: bool = False,
        offset_limit: Optional[str] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
        threshold: Optional[float] = None,
        weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Search for similar items in the Cosmos DB collection.

        Args:
            vectors: Query embedding vector.
            limit: Maximum number of results to return.
            search_type: The type of search to perform. Valid options are:
                ``[vector, full_text_search, full_text_ranking, hybrid]``.

                Behavioural variants are controlled by the optional
                ``threshold`` and ``weights`` arguments rather than separate
                search types: pass ``threshold`` alongside ``vector`` /
                ``hybrid`` to post-filter by per-row similarity score, and pass
                ``weights`` alongside ``hybrid`` to run server-side weighted
                RRF.
            return_with_vectors: Set to True to include vector embeddings in the search results.
                Only applicable for vector and hybrid search types.
            offset_limit: Optional ``OFFSET <int> LIMIT <int>`` clause for pagination.
                Validated against this exact format; rejected otherwise.
            full_text_rank_filter: Optional list of full text rank filters.
                Required for ``full_text_ranking`` and all hybrid search types.
            projection_mapping: Optional mapping for projecting specific fields.
            where: Optional raw SQL ``WHERE`` clause body (without the ``WHERE``
                keyword) for filtering results.

                .. warning::
                    This value is interpolated into the SQL query verbatim and is
                    NOT parameterised. Do not pass untrusted input. Use
                    parameterised filters or sanitise/validate strictly upstream.
            threshold: Optional similarity / distance score threshold. Applied
                as a strict client-side filter on the projected
                ``SimilarityScore`` for vector-bearing search types
                (``vector`` and ``hybrid``); ignored for pure full-text
                searches. Comparison direction depends on the container's
                ``distanceFunction``:

                * ``cosine`` (default) and ``dotproduct`` produce a similarity
                  score where higher = more similar; items are kept when
                  ``score > threshold`` (strict).
                * ``euclidean`` produces a distance where lower = more similar;
                  items are kept when ``score < threshold`` (strict).

                Pass ``None`` (the default) to disable threshold filtering.
            weights: Optional list of per-component RRF weights for ``hybrid``
                search. When provided, the ORDER BY clause is rewritten as
                CosmosDB's server-side ``RRF(..., [w1, w2, ...])`` form.
                Length must equal ``len(full_text_rank_filter) + 1`` (one per
                text component plus one for the vector component). E.g.
                ``[0.3, 0.7]`` weights the full-text component at 0.3 and
                the vector component at 0.7.

        Backward compatibility:
            pre_filter (dict): Legacy kwarg accepted by the old ``_query`` method.
                ``pre_filter["where_clause"]`` maps to ``where``.
                ``pre_filter["limit_offset_clause"]`` maps to ``offset_limit``.

        Returns:
            A VectorStoreQueryResult containing the results of the search query.

        """
        # Backward-compat: support old pre_filter={"where_clause": ..., "limit_offset_clause": ...}
        pre_filter = kwargs.pop("pre_filter", None)
        if pre_filter:
            if where is None and pre_filter.get("where_clause"):
                where = pre_filter["where_clause"]
            if offset_limit is None and pre_filter.get("limit_offset_clause"):
                offset_limit = pre_filter["limit_offset_clause"]

        # Validate search arguments
        self._validate_search_args(
            search_type=search_type,
            vector=vectors,
            return_with_vectors=return_with_vectors,
            full_text_rank_filter=full_text_rank_filter,
            offset_limit=offset_limit,
            weights=weights,
            threshold=threshold,
        )

        # Construct the query
        constructed_query, parameters = self._construct_search_query(
            limit=limit,
            search_type=search_type,
            vector=vectors,
            full_text_rank_filter=full_text_rank_filter,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            return_with_vectors=return_with_vectors,
            where=where,
            weights=weights,
        )

        # Execute query and get results
        return self._execute_search_query(
            query=constructed_query,
            search_type=search_type,
            parameters=parameters,
            return_with_vectors=return_with_vectors,
            projection_mapping=projection_mapping,
            threshold=threshold,
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Only ``query.query_embedding`` and ``query.similarity_top_k`` from the
        ``VectorStoreQuery`` object are consumed. Other fields — including
        ``query.mode``, ``query.filters``, ``query.query_str``,
        ``query.doc_ids``, ``query.node_ids`` — are intentionally ignored;
        callers must select the search variant via the ``search_type`` keyword
        argument and pass any filtering through ``where`` / ``full_text_rank_filter``.

        Keyword Args:
            search_type: The type of search to perform. Valid options are:
                ``[vector, full_text_search, full_text_ranking, hybrid]``.
            return_with_vectors: Set to True to include vector embeddings in the search results.
                Only applicable for vector and hybrid search types.
            offset_limit: Optional ``OFFSET <int> LIMIT <int>`` clause for pagination.
            full_text_rank_filter: Optional list of full text rank filters.
            projection_mapping: Optional mapping for projecting specific fields.
            where: Optional raw SQL WHERE clause body for filtering results
                (interpolated verbatim — do not pass untrusted input).
            threshold: Optional similarity / distance score threshold. Applied
                as a strict client-side filter for vector-bearing search types
                (``vector`` and ``hybrid``).
            weights: Optional list of per-component RRF weights for ``hybrid``
                search (server-side weighted RRF).

        Returns:
            A VectorStoreQueryResult containing the results of the query.

        """
        return self._search_query(
            vectors=query.query_embedding,
            limit=query.similarity_top_k,
            **kwargs,
        )
