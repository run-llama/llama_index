"""Alibaba Cloud OpenSearch Vector Store."""

import json
import logging
import asyncio
from typing import Any, List, Dict, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    FilterOperator,
    FilterCondition,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_TEXT_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

try:
    from alibabacloud_ha3engine_vector import client, models
except ImportError:
    raise ImportError(
        "`alibabacloud_ha3engine_vector` package not found, "
        "please run `pip install alibabacloud_ha3engine_vector`"
    )

DEFAULT_BATCH_SIZE = 100
logger = logging.getLogger(__name__)


def _to_ha3_filter_operator(op: FilterOperator) -> str:
    """Convert FilterOperator to Alibaba Cloud OpenSearch filter operator."""
    if op == FilterOperator.EQ:
        return "="
    elif (
        op == FilterOperator.IN
        or op == FilterOperator.NIN
        or op == FilterOperator.TEXT_MATCH
        or op == FilterOperator.CONTAINS
    ):
        raise ValueError(
            "Alibaba Cloud OpenSearch not support filter operator: `in/nin/text_match/contains` yet"
        )
    else:
        return op.value


def _to_ha3_engine_filter(
    standard_filters: Optional[MetadataFilters] = None,
) -> str:
    """Convert from standard filter to Alibaba Cloud OpenSearch filter spec."""
    if standard_filters is None:
        return ""

    filters = []
    for filter in standard_filters.filters:
        if isinstance(filter.value, str):
            value = f'"{filter.value}"'
        else:
            value = f"{filter.value}"
        filters.append(
            f"{filter.key} {_to_ha3_filter_operator(filter.operator)} {value}"
        )
    if standard_filters.condition == FilterCondition.AND:
        return " AND ".join(filters)
    elif standard_filters.condition == FilterCondition.OR:
        return " OR ".join(filters)
    else:
        raise ValueError(f"Unknown filter condition {standard_filters.condition}")


class AlibabaCloudOpenSearchConfig:
    """
    `Alibaba Cloud Opensearch` client configuration.

    Attribute:
        endpoint (str) : The endpoint of opensearch instance, You can find it
         from the console of Alibaba Cloud OpenSearch.
        instance_id (str) : The identify of opensearch instance, You can find
         it from the console of Alibaba Cloud OpenSearch.
        username (str) : The username specified when purchasing the instance.
        password (str) : The password specified when purchasing the instance,
          After the instance is created, you can modify it on the console.
        tablename (str): The table name specified during instance configuration.
        namespace (str) : The instance data will be partitioned based on the "namespace"
         field. If the namespace is enabled, you need to specify the namespace field
         name during initialization, Otherwise, the queries cannot be executed
         correctly.
        field_mapping (dict[str, str]): The field mapping between llamaindex meta field
          and OpenSearch table filed name. OpenSearch has some rules for the field name,
          when the meta field name break the rules, can map to another name.
        output_fields (list[str]): Specify the field list returned when searching OpenSearch.
        id_field (str): The primary key field name in OpenSearch, default is `id`.
        embedding_field (list[float]): The field name which stored the embedding.
        text_field: The name of the field that stores the key text.
        search_config (dict, optional): The configuration used for searching the OpenSearch.

    """

    def __init__(
        self,
        endpoint: str,
        instance_id: str,
        username: str,
        password: str,
        table_name: str,
        namespace: str = "",
        field_mapping: Dict[str, str] = None,
        output_fields: Optional[List[str]] = None,
        id_field: str = "id",
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        text_field: str = DEFAULT_TEXT_KEY,
        search_config: dict = None,
    ) -> None:
        self.endpoint = endpoint
        self.instance_id = instance_id
        self.username = username
        self.password = password
        self.namespace = namespace
        self.table_name = table_name
        self.data_source_name = f"{self.instance_id}_{self.table_name}"
        self.field_mapping = field_mapping
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.search_config = search_config
        self.output_fields = output_fields

        if self.output_fields is None:
            self.output_fields = (
                list(self.field_mapping.values()) if self.field_mapping else []
            )
        if self.text_field not in self.output_fields:
            self.output_fields.append(self.text_field)

        self.inverse_field_mapping: Dict[str, str] = (
            {value: key for key, value in self.field_mapping.items()}
            if self.field_mapping
            else {}
        )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class AlibabaCloudOpenSearchStore(BasePydanticVectorStore):
    """
    The AlibabaCloud OpenSearch Vector Store.

    In this vector store we store the text, its embedding and its metadata
    in a OpenSearch table.

    In order to use this you need to have a instance and configure a table.
    See the following documentation for details:
    https://help.aliyun.com/zh/open-search/vector-search-edition/product-overview

    Args:
        config (AlibabaCloudOpenSearchConfig): The instance configuration

    Examples:
        `pip install llama-index-vector-stores-alibabacloud_opensearch`

        ```python
        from llama_index.vector_stores.alibabacloud_opensearch import (
            AlibabaCloudOpenSearchConfig,
            AlibabaCloudOpenSearchStore,
        )

        # Config
        config = AlibabaCloudOpenSearchConfig(
            endpoint="xxx",
            instance_id="ha-cn-******",
            username="****",
            password="****",
            table_name="your_table_name",
        )

        vector_store = AlibabaCloudOpenSearchStore(config)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    _client: Any = PrivateAttr()
    _config: AlibabaCloudOpenSearchConfig = PrivateAttr()

    def __init__(self, config: AlibabaCloudOpenSearchConfig) -> None:
        """Initialize params."""
        super().__init__()

        self._config = config
        self._client = client.Client(
            models.Config(
                endpoint=config.endpoint,
                instance_id=config.instance_id,
                access_user_name=config.username,
                access_pass_word=config.password,
            )
        )

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AlibabaCloudOpenSearchStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to vector store.

        Args:
            nodes (List[BaseNode]): list of nodes with embeddings

        """
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(nodes, **add_kwargs)
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add nodes with embedding to vector store.

        Args:
            nodes (List[BaseNode]): list of nodes with embeddings

        """
        for i in range(0, len(nodes), DEFAULT_BATCH_SIZE):
            docs = []
            for node in nodes[i:DEFAULT_BATCH_SIZE]:
                doc = {
                    self._config.id_field: node.node_id,
                    self._config.embedding_field: node.embedding,
                }
                if self._config.text_field:
                    doc[self._config.text_field] = node.get_text()

                meta_fields = node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=self.flat_metadata
                )

                if self._config.field_mapping:
                    for key, value in meta_fields.items():
                        doc[self._config.field_mapping.get(key, key)] = value
                else:
                    doc.update(meta_fields)
                docs.append(doc)

            try:
                await self._async_send_data("add", docs)
            except Exception as e:
                logging.error(f"Add to {self._config.instance_id} failed: {e}")
                raise RuntimeError(f"Fail to add docs, error:{e}")
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.adelete(ref_doc_id, **delete_kwargs)
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronously delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        filter = f"{DEFAULT_DOC_ID_KEY}='{ref_doc_id}'"
        request = models.FetchRequest(table_name=self._config.table_name, filter=filter)

        response = self._client.fetch(request)
        json_response = json.loads(response.body)
        err_msg = json_response.get("errorMsg", None)
        if err_msg:
            raise RuntimeError(f"Failed to query doc by {filter}: {err_msg}")

        docs = []
        for doc in json_response["result"]:
            docs.append({"id": doc["id"]})
        await self._async_send_data("delete", docs)

    async def _async_send_data(self, cmd: str, fields_list: List[dict]) -> None:
        """
        Asynchronously send data.

        Args:
            cmd (str): data operator, add: upsert the doc, delete: delete the doc
            fields_list (list[dict]): doc fields list

        """
        docs = []
        for fields in fields_list:
            docs.append({"cmd": cmd, "fields": fields})
        request = models.PushDocumentsRequest({}, docs)
        await self._client.push_documents_async(
            self._config.data_source_name, self._config.id_field, request
        )

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query vector store."""
        return asyncio.get_event_loop().run_until_complete(self.aquery(query, **kwargs))

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronously query vector store.
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(
                f"Alibaba Cloud OpenSearch does not support {query.mode} yet."
            )

        request = self._gen_query_request(query)
        response = await self._client.query_async(request)
        json_response = json.loads(response.body)
        logging.debug(f"query result: {json_response}")

        err_msg = json_response.get("errorMsg", None)
        if err_msg:
            raise RuntimeError(
                f"query doc from Alibaba Cloud OpenSearch instance:{self._config.instance_id} failed:"
                f"{err_msg}"
            )

        ids = []
        nodes = []
        similarities = []
        for doc in json_response["result"]:
            try:
                node = metadata_dict_to_node(
                    {
                        "_node_content": doc["fields"].get(
                            self._config.field_mapping.get(
                                "_node_content", "_node_content"
                            ),
                            None,
                        ),
                        "_node_type": doc["fields"].get(
                            self._config.field_mapping.get("_node_type", "_node_type"),
                            None,
                        ),
                    }
                )
            except Exception:
                text = doc["fields"][self._config.text_field]
                metadata = {
                    self._config.inverse_field_mapping.get(key, key): doc["fields"].get(
                        key
                    )
                    for key in self._config.output_fields
                }
                node = TextNode(id_=doc["id"], text=text, metadata=metadata)

            ids.append(doc["id"])
            nodes.append(node)
            similarities.append(doc["score"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _gen_query_request(self, query: VectorStoreQuery) -> models.QueryRequest:
        """
        Generate the OpenSearch query request.

        Args:
            query (VectorStoreQuery): The vector store query

        Return:
            OpenSearch query request

        """
        filter = _to_ha3_engine_filter(query.filters)
        request = models.QueryRequest(
            table_name=self._config.table_name,
            namespace=self._config.namespace,
            vector=query.query_embedding,
            top_k=query.similarity_top_k,
            filter=filter,
            include_vector=True,
            output_fields=self._config.output_fields,
        )

        if self._config.search_config:
            request.order = self._config.search_config.get("order", "ASC")
            score_threshold: float = self._config.search_config.get(
                "score_threshold", None
            )
            if score_threshold is not None:
                request.score_threshold = score_threshold
            search_params = self._config.search_config.get("search_params", None)
            if search_params is not None:
                request.search_params = json.dumps(search_params)
        return request
