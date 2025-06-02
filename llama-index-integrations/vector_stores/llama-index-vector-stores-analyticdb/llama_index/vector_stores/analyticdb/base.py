"""AnalyticDB vector store."""

import logging
import json
from typing import Any, List, Union

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)


_logger = logging.getLogger(__name__)
_import_err_msg = (
    "`alibabacloud_gpdb20160503` and `alibabacloud_tea_openapi` packages not found, "
    "please run `pip install alibabacloud_gpdb20160503 alibabacloud_tea_openapi`"
)

OPERATOR_MAP = {
    FilterOperator.EQ: "=",
    FilterOperator.GT: ">",
    FilterOperator.LT: "<",
    FilterOperator.NE: "!=",
    FilterOperator.GTE: ">=",
    FilterOperator.LTE: "<=",
    FilterOperator.IN: "IN",
    FilterOperator.NIN: "NOT IN",
    FilterOperator.CONTAINS: "@>",
}


def _build_filter_clause(filter_: MetadataFilter) -> str:
    adb_operator = OPERATOR_MAP.get(filter_.operator)
    if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
        return f"metadata_->>'{filter_.key}' {adb_operator} {tuple(filter_.value)}"
    elif filter_.operator == FilterOperator.CONTAINS:
        return (
            f"metadata_::jsonb->'{filter_.key}' {adb_operator} '[\"{filter_.value}\"]'"
        )
    else:
        return f"metadata_->>'{filter_.key}' {adb_operator} '{filter_.value}'"


def _recursively_parse_adb_filter(filters: MetadataFilters) -> Union[str, None]:
    if not filters:
        return None
    return f" {filters.condition} ".join(
        [
            (
                _build_filter_clause(filter_)
                if isinstance(filter_, MetadataFilter)
                else f"({_recursively_parse_adb_filter(filter_)})"
            )
            for filter_ in filters.filters
        ]
    )


class AnalyticDBVectorStore(BasePydanticVectorStore):
    """
    AnalyticDB vector store.

    In this vector store, embeddings and docs are stored within a
    single table.

    During query time, the index uses AnalyticDB to query for the top
    k most similar nodes.

    Args:
        region_id: str
        instance_id: str
        account: str
        account_password: str
        namespace: str
        namespace_password: str
        embedding_dimension: int
        metrics: str
        collection: str

    """

    stores_text: bool = True
    flat_metadata: bool = False

    region_id: str
    instance_id: str
    account: str
    account_password: str
    namespace: str
    namespace_password: str
    embedding_dimension: int
    metrics: str
    collection: str

    _client: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        client: Any,
        region_id: str,
        instance_id: str,
        account: str,
        account_password: str,
        namespace: str = "llama_index",
        collection: str = "llama_collection",
        namespace_password: str = None,
        embedding_dimension: int = 1536,
        metrics: str = "cosine",
    ):
        try:
            from alibabacloud_gpdb20160503.client import Client
        except ImportError:
            raise ImportError(_import_err_msg)

        if client is not None:
            if not isinstance(client, Client):
                raise ValueError(
                    "client must be of type alibabacloud_gpdb20160503.client.Client"
                )
        else:
            raise ValueError("client not specified")
        if not namespace_password:
            namespace_password = account_password
        super().__init__(
            region_id=region_id,
            instance_id=instance_id,
            account=account,
            account_password=account_password,
            namespace=namespace,
            collection=collection,
            namespace_password=namespace_password,
            embedding_dimension=embedding_dimension,
            metrics=metrics,
        )
        self._client = client

    @classmethod
    def _initialize_client(
        cls,
        access_key_id: str,
        access_key_secret: str,
        region_id: str,
        read_timeout: int = 60000,
    ) -> Any:
        """
        Initialize ADB client.
        """
        try:
            from alibabacloud_gpdb20160503.client import Client
            from alibabacloud_tea_openapi import models as open_api_models
        except ImportError:
            raise ImportError(_import_err_msg)

        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=region_id,
            read_timeout=read_timeout,
            user_agent="llama-index",
        )
        return Client(config)

    @classmethod
    def from_params(
        cls,
        access_key_id: str,
        access_key_secret: str,
        region_id: str,
        instance_id: str,
        account: str,
        account_password: str,
        namespace: str = "llama_index",
        collection: str = "llama_collection",
        namespace_password: str = None,
        embedding_dimension: int = 1536,
        metrics: str = "cosine",
        read_timeout: int = 60000,
    ) -> "AnalyticDBVectorStore":
        client = cls._initialize_client(
            access_key_id, access_key_secret, region_id, read_timeout
        )
        return cls(
            client=client,
            region_id=region_id,
            instance_id=instance_id,
            account=account,
            account_password=account_password,
            namespace=namespace,
            collection=collection,
            namespace_password=namespace_password,
            embedding_dimension=embedding_dimension,
            metrics=metrics,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AnalyticDBVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models

        self._initialize()
        ids = []
        rows: List[gpdb_20160503_models.UpsertCollectionDataRequestRows] = []
        for node in nodes:
            ids.append(node.node_id)
            node_metadata_dict = node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            )
            metadata = {
                "node_id": node.node_id,
                "ref_doc_id": node.ref_doc_id,
                "content": node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata_": json.dumps(node_metadata_dict),
            }
            rows.append(
                gpdb_20160503_models.UpsertCollectionDataRequestRows(
                    vector=node.get_embedding(),
                    metadata=metadata,
                )
            )
        _logger.debug("adding nodes to vector store...")
        request = gpdb_20160503_models.UpsertCollectionDataRequest(
            dbinstance_id=self.instance_id,
            region_id=self.region_id,
            namespace=self.namespace,
            namespace_password=self.namespace_password,
            collection=self.collection,
            rows=rows,
        )
        response = self._client.upsert_collection_data(request)
        _logger.info(
            f"successfully adding nodes to vector store, size: {len(nodes)},"
            f"response body:{response.body}"
        )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete a node from the vector store.

        Args:
            ref_doc_id: str: the doc_id of the document to delete.

        """
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models

        self._initialize()
        collection_data = '{"ref_doc_id": ["%s"]}' % ref_doc_id
        request = gpdb_20160503_models.DeleteCollectionDataRequest(
            dbinstance_id=self.instance_id,
            region_id=self.region_id,
            namespace=self.namespace,
            namespace_password=self.namespace_password,
            collection=self.collection,
            collection_data=collection_data,
        )
        _logger.debug(f"deleting nodes from vector store of ref_doc_id: {ref_doc_id}")
        response = self._client.delete_collection_data(request)
        _logger.info(
            f"successfully delete nodes from vector store, response body: {response.body}"
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store for top k most similar nodes.

        Args:
            query: VectorStoreQuery: the query to execute.

        Returns:
            VectorStoreQueryResult: the result of the query.

        """
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models

        self._initialize()
        vector = (
            query.query_embedding
            if query.mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID)
            else None
        )
        content = (
            query.query_str
            if query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            else None
        )
        request = gpdb_20160503_models.QueryCollectionDataRequest(
            dbinstance_id=self.instance_id,
            region_id=self.region_id,
            namespace=self.namespace,
            namespace_password=self.namespace_password,
            collection=self.collection,
            include_values=kwargs.pop("include_values", True),
            metrics=self.metrics,
            vector=vector,
            content=content,
            top_k=query.similarity_top_k,
            filter=_recursively_parse_adb_filter(query.filters),
        )
        response = self._client.query_collection_data(request)
        nodes = []
        similarities = []
        ids = []
        for match in response.body.matches.match:
            node = metadata_dict_to_node(
                json.loads(match.metadata.get("metadata_")),
                match.metadata.get("content"),
            )
            nodes.append(node)
            similarities.append(match.score)
            ids.append(match.metadata.get("node_id"))
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def delete_collection(self):
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models

        request = gpdb_20160503_models.DeleteCollectionRequest(
            dbinstance_id=self.instance_id,
            region_id=self.region_id,
            namespace=self.namespace,
            namespace_password=self.namespace_password,
            collection=self.collection,
        )
        self._client.delete_collection(request)
        _logger.debug(f"collection {self.collection} deleted")

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._initialize_vector_database()
            self._create_namespace_if_not_exists()
            self._create_collection_if_not_exists()
            self._is_initialized = True

    def _initialize_vector_database(self):
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models

        request = gpdb_20160503_models.InitVectorDatabaseRequest(
            dbinstance_id=self.instance_id,
            region_id=self.region_id,
            manager_account=self.account,
            manager_account_password=self.account_password,
        )
        response = self._client.init_vector_database(request)
        _logger.debug(
            f"successfully initialize vector database, response body:{response.body}"
        )

    def _create_namespace_if_not_exists(self):
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models
        from Tea.exceptions import TeaException

        try:
            request = gpdb_20160503_models.DescribeNamespaceRequest(
                dbinstance_id=self.instance_id,
                region_id=self.region_id,
                namespace=self.namespace,
                manager_account=self.account,
                manager_account_password=self.account_password,
            )
            self._client.describe_namespace(request)
            _logger.debug(f"namespace {self.namespace} already exists")
        except TeaException as e:
            if e.statusCode == 404:
                _logger.debug(f"namespace {self.namespace} does not exist, creating...")
                request = gpdb_20160503_models.CreateNamespaceRequest(
                    dbinstance_id=self.instance_id,
                    region_id=self.region_id,
                    manager_account=self.account,
                    manager_account_password=self.account_password,
                    namespace=self.namespace,
                    namespace_password=self.namespace_password,
                )
                self._client.create_namespace(request)
                _logger.debug(f"namespace {self.namespace} created")
            else:
                raise ValueError(f"failed to create namespace {self.namespace}: {e}")

    def _create_collection_if_not_exists(self):
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models
        from Tea.exceptions import TeaException

        try:
            request = gpdb_20160503_models.DescribeCollectionRequest(
                dbinstance_id=self.instance_id,
                region_id=self.region_id,
                namespace=self.namespace,
                namespace_password=self.namespace_password,
                collection=self.collection,
            )
            self._client.describe_collection(request)
            _logger.debug(f"collection {self.collection} already exists")
        except TeaException as e:
            if e.statusCode == 404:
                _logger.debug(
                    f"collection {self.namespace} does not exist, creating..."
                )
                metadata = '{"node_id":"text","ref_doc_id":"text","content":"text","metadata_":"jsonb"}'
                full_text_retrieval_fields = "content"
                request = gpdb_20160503_models.CreateCollectionRequest(
                    dbinstance_id=self.instance_id,
                    region_id=self.region_id,
                    manager_account=self.account,
                    manager_account_password=self.account_password,
                    namespace=self.namespace,
                    collection=self.collection,
                    dimension=self.embedding_dimension,
                    metrics=self.metrics,
                    metadata=metadata,
                    full_text_retrieval_fields=full_text_retrieval_fields,
                )
                self._client.create_collection(request)
                _logger.debug(f"collection {self.namespace} created")
            else:
                raise ValueError(f"failed to create collection {self.collection}: {e}")
