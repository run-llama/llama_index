"""
Google BigQuery Vector Search.

BigQuery Vector Search is a fully managed feature of BigQuery that enables fast,
scalable similarity search over high-dimensional embeddings using approximate
nearest neighbor methods. For more information visit:
https://cloud.google.com/bigquery/docs/vector-search-intro
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from google.auth import credentials
from google.cloud import bigquery
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pydantic import BaseModel

from llama_index.vector_stores.bigquery.utils import build_where_clause_and_params


_logger = logging.getLogger(__name__)


class _BigQueryRow(BaseModel):
    node_id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class DistanceType(str, Enum):
    EUCLIDEAN = "EUCLIDEAN"
    COSINE = "COSINE"
    DOT_PRODUCT = "DOT_PRODUCT"


class BigQueryVectorStore(BasePydanticVectorStore):
    """
    Vector store index using Google BigQuery.

    Provides integration with BigQuery for storing and querying vector embeddings.
    For more information, visit: https://cloud.google.com/bigquery/docs/vector-search-intro

    Required IAM Permissions:
        - `roles/bigquery.dataOwner` (BigQuery Data Owner)
        - `roles/bigquery.dataEditor` (BigQuery Data Editor)

    Examples:
        `pip install llama-index-vector-stores-bigquery`

        ```python
        from google.cloud.bigquery import Client
        from llama_index.vector_stores.bigquery import BigQueryVectorStore

        client = Client()

        vector_store = BigQueryVectorStore(
            table_id="my_bigquery_table",
            dataset_id="my_bigquery_dataset",
            bigquery_client=client,
        )
        ```

    """

    stores_text: bool = True
    distance_type: DistanceType = DistanceType.EUCLIDEAN

    _table: bigquery.Table = PrivateAttr()
    _dataset: bigquery.Dataset = PrivateAttr()
    _client: bigquery.Client = PrivateAttr()
    _full_table_id: str = PrivateAttr()

    def __init__(
        self,
        table_id: str,
        dataset_id: str,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        distance_type: Optional[DistanceType] = DistanceType.EUCLIDEAN,
        auth_credentials: Optional[credentials.Credentials] = None,
        bigquery_client: Optional[bigquery.Client] = None,
        **kwargs: Any,
    ):
        """
        Initialize a BigQuery Vector store.

        If a `bigquery_client` is provided, it will be used directly. Otherwise, a client will be initialized using
        the optional `project_id`, `region`, and/or `auth_credentials`. If none are provided, default credentials
        will be used. For details on authentication, visit:
        https://googleapis.dev/python/google-api-core/latest/auth.html

        Args:
            table_id: The ID of the BigQuery table to use for vector storage.
            dataset_id: The ID of the dataset containing the table.
            project_id: The GCP project ID. If not provided, it will be inferred from the client or environment.
            region: Optionally specify a default location for datasets / tables.
            distance_type: Optionally specify a distance type to use `EUCLIDEAN`, `COSINE`, or `DOT_PRODUCT`.
            auth_credentials: Optional credentials object used to authenticate with BigQuery.
            bigquery_client: An existing BigQuery client instance. If not provided, one will be created.
            **kwargs: Additional keyword arguments passed to the parent class.

        """
        super().__init__(
            **kwargs,
        )

        self._client: bigquery.Client = bigquery_client or self._initialize_client(
            project_id, region, auth_credentials
        )
        self._dataset: bigquery.Dataset = self._create_dataset_if_not_exists(dataset_id)
        self._table: bigquery.Table = self._create_table_if_not_exists(table_id)
        self._full_table_id: str = (
            f"{self._client.project}.{self._dataset.dataset_id}.{self._table.table_id}"
        )
        self.distance_type: DistanceType = DistanceType(distance_type)

    @classmethod
    def from_params(
        cls,
        table_id: str,
        dataset_id: str,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        distance_type: Optional[DistanceType] = DistanceType.EUCLIDEAN,
        auth_credentials: Optional[credentials.Credentials] = None,
        bigquery_client: Optional[bigquery.Client] = None,
    ) -> "BigQueryVectorStore":
        """
        Initialize a BigQuery Vector store.

        Args:
            table_id: The ID of the BigQuery table to use for vector storage.
            dataset_id: The ID of the dataset containing the table.
            project_id: The GCP project ID. If not provided, it will be inferred from the client or environment.
            region: Optionally specify a default location for datasets / tables.
            distance_type: Optionally specify a distance type to use `EUCLIDEAN`, `COSINE`, or `DOT_PRODUCT`.
            auth_credentials: Optional credentials object used to authenticate with BigQuery.
            bigquery_client: An existing BigQuery client instance. If not provided, one will be created.

        Returns:
            BigQueryVectorStore

        """
        return cls(
            table_id=table_id,
            dataset_id=dataset_id,
            project_id=project_id,
            region=region,
            distance_type=distance_type,
            auth_credentials=auth_credentials,
            bigquery_client=bigquery_client,
        )

    @property
    def client(self) -> Union[bigquery.Client, None]:
        """Return the BigQuery client."""
        if not self._client:
            return None
        return self._client

    @staticmethod
    def _initialize_client(
        project_id: Union[str, None],
        region: Union[str, None],
        auth_credentials: Union[credentials.Credentials, None],
    ) -> bigquery.Client:
        """
        Initialize a new BigQuery client using the provided `project_id`, `region` and/or `auth_credentials`.
        Defaults will be used in place of missing arguments. For details on authentication, see:
        https://googleapis.dev/python/google-api-core/latest/auth.html

        Args:
            project_id: GCP project ID for the new client, or None to use default project resolution.
            region: GCP region for the new client, or None to use default region.
            auth_credentials: Credentials to authenticate the new client, or None to use default credentials.

        Returns:
            An initialized BigQuery client.

        """
        return bigquery.Client(
            project=project_id or None,
            location=region or None,
            credentials=auth_credentials or None,
        )

    @staticmethod
    def _bigquery_row_to_node(row: _BigQueryRow) -> BaseNode:
        """
        Convert a BigQuery row to a BaseNode object.

        Args:
            row: A row retrieved from BigQuery containing node_id, text,
                metadata, embedding, and optional distance.

        Returns:
            Node object.

        """
        node_id: str = row.node_id
        text: str = row.text
        metadata: Dict[str, Any] = row.metadata
        embedding: List[float] = row.embedding
        _: Union[float, None] = row.distance

        try:
            node = metadata_dict_to_node(metadata)
            node.set_content(text)
            node.embedding = embedding
        except (ValueError, TypeError) as e:
            node = TextNode(
                id_=node_id,
                text=text,
                metadata=metadata,
                embedding=embedding,
            )
            _logger.warning(
                f"Failed to construct node {node_id} from metadata. Falling back to manual construction. Error: {e}"
            )

        return node

    def _create_dataset_if_not_exists(self, dataset_id: str) -> bigquery.Dataset:
        """
        Create a BigQuery dataset if it does not already exist.

        For more details on creating datasets, visit:
        https://cloud.google.com/bigquery/docs/datasets#create-dataset

        Args:
            dataset_id: The ID of the dataset to create.

        Returns:
            Dataset ID.

        """
        dataset_ref = bigquery.dataset.DatasetReference(
            project=self._client.project, dataset_id=dataset_id
        )

        return self._client.create_dataset(dataset_ref, exists_ok=True)

    def _create_table_if_not_exists(self, table_id) -> bigquery.Table:
        """
        Create a BigQuery table if it does not already exist.

        For more information on creating tables, visit:
        https://cloud.google.com/bigquery/docs/tables#create-table

        Args:
            table_id: The ID of the table to create.

        Returns:
            BigQuery table instance.

        """
        schema = [
            bigquery.SchemaField("node_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("metadata", "JSON"),
            bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
        ]

        table_ref = bigquery.TableReference.from_string(
            f"{self._client.project}.{self._dataset.dataset_id}.{table_id}"
        )
        to_create = bigquery.Table(table_ref, schema=schema)

        return self._client.create_table(to_create, exists_ok=True)

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List of nodes with embeddings.

        Returns:
            List of node IDs that were added.

        """
        node_ids: List[str] = []
        json_records: List[Dict[str, Any]] = []

        for node in nodes:
            record = {
                "node_id": node.node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "embedding": node.get_embedding(),
                "metadata": node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=False
                ),
            }
            node_ids.append(node.node_id)
            json_records.append(record)

        job_config = bigquery.LoadJobConfig(schema=self._table.schema)
        job = self._client.load_table_from_json(
            json_rows=json_records, destination=self._table, job_config=job_config
        )
        job.result()

        return node_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id : The doc_id of the document to delete.

        """
        query = f"""
        DELETE FROM `{self._full_table_id}`
        WHERE  SAFE.JSON_VALUE(metadata, '$."doc_id"') = @to_delete;
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    name="to_delete", type_="STRING", value=ref_doc_id
                ),
            ]
        )

        self._client.query_and_wait(query, job_config=job_config)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store using BigQuery's VECTOR_SEARCH to retrieve the top-k most similar nodes.

        When `MetadataFilters` are provided and the table is indexed on relevant columns, BigQuery attempts to optimize
        the search with pre-filtering before nearest neighbor search. If filters don't align with an index,
        post-filtering is applied after similarity search, potentially returning fewer than `similarity_top_k results`.
        Consider increasing `similarity_top_k` when post-filtering is expected.

        For more information on pre-filtering and post-filtering, see:
        https://cloud.google.com/bigquery/docs/vector-index#pre-filters_and_post-filters

        Assumes embeddings are normalized for similarity scoring.

        Args:
            query: Contains the query embedding, similarity_top_k value, and optional metadata filters.

        Returns:
            VectorStoreQueryResult

        """
        where_clause, query_params = build_where_clause_and_params(
            filters=query.filters, node_ids=query.node_ids
        )

        base_table_query = f"""
        SELECT
            node_id,
            text,
            metadata,
            embedding
        FROM `{self._full_table_id}`
        """

        if where_clause:
            base_table_query += f" WHERE {where_clause}"

        query_table_query = f"SELECT {query.query_embedding} AS input_embedding"

        vector_search_query = f"""
        SELECT  base.node_id   AS node_id,
                base.text      AS text,
                base.metadata  AS metadata,
                base.embedding AS embedding,
                distance
        FROM
            VECTOR_SEARCH(
                ({base_table_query}), 'embedding',
                ({query_table_query}), 'input_embedding',
                top_k => @top_k,
                distance_type => @distance_type
        );
        """

        query_params.extend(
            [
                bigquery.ScalarQueryParameter(
                    "top_k", type_="INTEGER", value=query.similarity_top_k
                ),
                bigquery.ScalarQueryParameter(
                    "distance_type", type_="STRING", value=self.distance_type
                ),
            ]
        )
        job_config = bigquery.QueryJobConfig(
            query_parameters=query_params,
        )
        rows: bigquery.table.RowIterator = self._client.query_and_wait(
            vector_search_query, job_config=job_config
        )

        top_k_nodes: List[BaseNode] = []
        top_k_scores: List[float] = []
        top_k_ids: List[str] = []

        for record in rows:
            row = _BigQueryRow(
                node_id=record.node_id,
                text=record.text,
                metadata=record.metadata,
                embedding=record.embedding,
                distance=record.distance,
            )
            node = self._bigquery_row_to_node(row)
            node_id = record.node_id
            # Assumes embeddings are normalized.
            score = (
                1 / (1 + record.distance)
                if self.distance_type == DistanceType.EUCLIDEAN
                else (1 + record.distance) / 2
            )

            top_k_nodes.append(node)
            top_k_scores.append(score)
            top_k_ids.append(node_id)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Retrieve nodes from BigQuery using node IDs, metadata filters, or both.

        If both `node_ids` and `filters` are provided, only nodes that satisfy
        both conditions will be returned.

        Args:
            node_ids: Optional list of node IDs for retrieval.
            filters : Optional MetadataFilters filters for retrieval.

        Returns:
            A list of matching nodes.

        Raises:
            ValueError: If neither `node_ids` nor `filters` is provided.

        """
        if not (node_ids or filters):
            raise ValueError(
                "get_nodes requires at least one filtering parameter: "
                "'node_ids', 'filters', or both. Received neither."
            )

        where_clause, query_params = build_where_clause_and_params(node_ids, filters)

        query = f"""
        SELECT  node_id,
                text,
                embedding,
                metadata
        FROM    `{self._full_table_id}`
        WHERE   {where_clause};
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=query_params,
        )
        rows: bigquery.table.RowIterator = self._client.query_and_wait(
            query, job_config=job_config
        )

        nodes: List[BaseNode] = []
        for record in rows:
            row = _BigQueryRow(
                node_id=record.node_id,
                text=record.text,
                metadata=record.metadata,
                embedding=record.embedding,
                distance=record.distance,
            )
            node = self._bigquery_row_to_node(row)

            nodes.append(node)

        return nodes

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes from BigQuery based on node IDs, metadata filters, or both.

        If both `node_ids` and `filters` are provided, only nodes matching both
        criteria will be deleted.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters : Optional MetadataFilters filters for deletion.

        Raises:
            ValueError: If neither `node_ids` nor `filters` are provided.

        """
        if not (node_ids or filters):
            raise ValueError(
                "delete_nodes requires at least one filtering parameter: "
                "'node_ids', 'filters', or both. Received neither."
            )

        where_clause, query_params = build_where_clause_and_params(node_ids, filters)

        query = f"""
        DELETE FROM `{self._full_table_id}`
        WHERE {where_clause};
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=query_params,
        )
        self._client.query_and_wait(query, job_config=job_config)

    def clear(self) -> None:
        """
        Clears the index.

        This truncates the underlying table in BigQuery.
        """
        query = f"""TRUNCATE TABLE `{self._full_table_id}`;"""
        self._client.query_and_wait(query)
