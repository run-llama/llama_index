"""Weaviate Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, Dict, List, Optional, Union, cast
from uuid import uuid4

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import DEFAULT_TEXT_KEY
from llama_index.vector_stores.weaviate.utils import (
    get_data_object,
    aclass_schema_exists,
    class_schema_exists,
    create_default_schema,
    acreate_default_schema,
    get_node_similarity,
    to_node,
)

import weaviate
from weaviate import Client
import weaviate.classes as wvc

_logger = logging.getLogger(__name__)


def _transform_weaviate_filter_condition(condition: str) -> str:
    """Translate standard metadata filter op to Chroma specific spec."""
    if condition == "and":
        return wvc.query.Filter.all_of
    elif condition == "or":
        return wvc.query.Filter.any_of
    else:
        raise ValueError(f"Filter condition {condition} not supported")


def _transform_weaviate_filter_operator(operator: str) -> str:
    """Translate standard metadata filter operator to Weaviate specific spec."""
    if operator == "!=":
        return "not_equal"
    elif operator == "==":
        return "equal"
    elif operator == ">":
        return "greater_than"
    elif operator == "<":
        return "less_than"
    elif operator == ">=":
        return "greater_or_equal"
    elif operator == "<=":
        return "less_or_equal"
    elif operator == "any":
        return "contains_any"
    elif operator == "all":
        return "contains_all"
    elif operator == "is_empty":
        return "is_none"
    else:
        raise ValueError(f"Filter operator {operator} not supported")


def _to_weaviate_filter(
    standard_filters: MetadataFilters,
) -> Union[wvc.query.Filter, List[wvc.query.Filter]]:
    filters_list = []
    condition = standard_filters.condition or "and"
    condition = _transform_weaviate_filter_condition(condition)

    if standard_filters.filters:
        for filter in standard_filters.filters:
            if isinstance(filter, MetadataFilters):
                filters_list.append(_to_weaviate_filter(filter))
                continue

            property_filter = getattr(
                wvc.query.Filter.by_property(filter.key),
                _transform_weaviate_filter_operator(filter.operator),
            )
            value = filter.value
            # IS_EMPTY does not take a value (expected to be set to None), but when using IsNull with Weaviate, a
            # boolean is expected (True meaning the value actually being null / not set / empty)
            if filter.operator == "is_empty":
                value = True
            filters_list.append(property_filter(value))
    else:
        return {}

    if len(filters_list) == 1:
        # If there is only one filter, return it directly
        return filters_list[0]

    return condition(filters_list)


class WeaviateVectorStore(BasePydanticVectorStore):
    """Weaviate vector store.

    In this vector store, embeddings and docs are stored within a
    Weaviate collection.

    During query time, the index uses Weaviate to query for the top
    k most similar nodes.

    Args:
        weaviate_client (weaviate.Client): WeaviateClient
            instance from `weaviate-client` package
        index_name (Optional[str]): name for Weaviate classes

    Examples:
        `pip install llama-index-vector-stores-weaviate`

        ```python
        import weaviate

        resource_owner_config = weaviate.AuthClientPassword(
            username="<username>",
            password="<password>",
        )
        client = weaviate.Client(
            "https://llama-test-ezjahb4m.weaviate.network",
            auth_client_secret=resource_owner_config,
        )

        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name="LlamaIndex"
        )
        ```
    """

    stores_text: bool = True

    index_name: str
    url: Optional[str]
    text_key: str
    auth_config: Dict[str, Any] = Field(default_factory=dict)
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _client: weaviate.WeaviateClient = PrivateAttr()
    _aclient: weaviate.WeaviateAsyncClient = PrivateAttr()

    def __init__(
        self,
        weaviate_client: Optional[Any] = None,
        weaviate_aclient: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        index_name: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        auth_config: Optional[Any] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # validate class prefix starts with a capital letter
        if class_prefix is not None:
            _logger.warning("class_prefix is deprecated, please use index_name")
            # legacy, kept for backward compatibility
            index_name = f"{class_prefix}_Node"

        index_name = index_name or f"LlamaIndex_{uuid4().hex}"
        if not index_name[0].isupper():
            raise ValueError(
                "Index name must start with a capital letter, e.g. 'LlamaIndex'"
            )

        super().__init__(
            url=url,
            index_name=index_name,
            text_key=text_key,
            auth_config=auth_config.__dict__ if auth_config else {},
            client_kwargs=client_kwargs or {},
        )

        if weaviate_client is None and weaviate_aclient is None:
            if isinstance(auth_config, dict):
                auth_config = weaviate.auth.AuthApiKey(auth_config)

            client_kwargs = client_kwargs or {}
            self._client = weaviate.WeaviateClient(
                auth_client_secret=auth_config, **client_kwargs
            )
        if weaviate_client:
            self._client = cast(weaviate.WeaviateClient, weaviate_client)
        else:
            self._client = None
        if weaviate_aclient:
            self._aclient = cast(weaviate.WeaviateAsyncClient, weaviate_aclient)
        else:
            self._aclient = None

        # create default schema if does not exist
        if self._client is not None:
            if not class_schema_exists(self._client, index_name):
                create_default_schema(self._client, index_name)
            self._collection_initialized = True
        else:
            #  need to do lazy init for async clients
            self._collection_initialized = False

    @classmethod
    def from_params(
        cls,
        url: str,
        auth_config: Any,
        index_name: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "WeaviateVectorStore":
        """Create WeaviateVectorStore from config."""
        client_kwargs = client_kwargs or {}
        weaviate_client = Client(
            url=url, auth_client_secret=auth_config, **client_kwargs
        )
        return cls(
            weaviate_client=weaviate_client,
            url=url,
            auth_config=auth_config.__dict__,
            client_kwargs=client_kwargs,
            index_name=index_name,
            text_key=text_key,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "WeaviateVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        ids = [r.node_id for r in nodes]

        with self._client.batch.dynamic() as batch:
            for node in nodes:
                data_object = get_data_object(node=node, text_key=self.text_key)
                batch.add_object(
                    collection=self.index_name,
                    properties=data_object.properties,
                    uuid=data_object.uuid,
                    vector=data_object.vector,
                )
        return ids

    async def async_add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if len(nodes) > 0 and not self._collection_initialized:
            if not await aclass_schema_exists(self._aclient, self.index_name):
                await acreate_default_schema(self._aclient, self.index_name)

        ids = [r.node_id for r in nodes]

        collection = self._aclient.collections.get(self.index_name)

        response = await collection.data.insert_many(
            [get_data_object(node=node, text_key=self.text_key) for node in nodes]
        )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        collection = self._client.collections.get(self.index_name)

        where_filter = wvc.query.Filter.by_property("ref_doc_id").equal(ref_doc_id)

        if "filter" in delete_kwargs and delete_kwargs["filter"] is not None:
            where_filter = where_filter & _to_weaviate_filter(delete_kwargs["filter"])

        collection.data.delete_many(where=where_filter)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        collection = self._aclient.collections.get(self.index_name)

        where_filter = wvc.query.Filter.by_property("ref_doc_id").equal(ref_doc_id)

        if "filter" in delete_kwargs and delete_kwargs["filter"] is not None:
            where_filter = where_filter & _to_weaviate_filter(delete_kwargs["filter"])

        result = await collection.data.delete_many(where=where_filter)

    def delete_index(self) -> None:
        """Delete the index associated with the client.

        Raises:
        - Exception: If the deletion fails, for some reason.
        """
        if not class_schema_exists(self._client, self.index_name):
            _logger.warning(
                f"Index '{self.index_name}' does not exist. No action taken."
            )
            return
        try:
            self._client.collections.delete(self.index_name)
            _logger.info(f"Successfully deleted index '{self.index_name}'.")
        except Exception as e:
            _logger.error(f"Failed to delete index '{self.index_name}': {e}")
            raise Exception(f"Failed to delete index '{self.index_name}': {e}")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        if not node_ids and not filters:
            return

        collection = self._client.collections.get(self.index_name)

        if node_ids:
            filter = wvc.query.Filter.by_id().contains_any(node_ids or [])

        if filters:
            if node_ids:
                filter = filter & _to_weaviate_filter(filters)
            else:
                filter = _to_weaviate_filter(filters)

        collection.data.delete_many(where=filter, **delete_kwargs)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        if not node_ids and not filters:
            return

        collection = self._aclient.collections.get(self.index_name)

        if node_ids:
            filter = wvc.query.Filter.by_id().contains_any(node_ids or [])

        if filters:
            if node_ids:
                filter = filter & _to_weaviate_filter(filters)
            else:
                filter = _to_weaviate_filter(filters)

        await collection.data.delete_many(where=filter, **delete_kwargs)

    def clear(self) -> None:
        """Clears index."""
        self.delete_index()

    async def aclear(self) -> None:
        """Delete the index associated with the client.

        Raises:
        - Exception: If the deletion fails, for some reason.
        """
        if not await aclass_schema_exists(self._aclient, self.index_name):
            _logger.warning(
                f"Index '{self.index_name}' does not exist. No action taken."
            )
            return
        try:
            await self._aclient.collections.delete(self.index_name)
            _logger.info(f"Successfully deleted index '{self.index_name}'.")
        except Exception as e:
            _logger.error(f"Failed to delete index '{self.index_name}': {e}")
            raise Exception(f"Failed to delete index '{self.index_name}': {e}")

    def get_query_parameters(self, query: VectorStoreQuery, **kwargs: Any):
        filters = None

        # list of documents to constrain search
        if query.doc_ids:
            filters = wvc.query.Filter.by_property("doc_id").contains_any(query.doc_ids)

        if query.node_ids:
            filters = wvc.query.Filter.by_property("id").contains_any(query.node_ids)

        return_metatada = wvc.query.MetadataQuery(distance=True, score=True)

        vector = query.query_embedding
        alpha = 1
        if query.mode == VectorStoreQueryMode.HYBRID:
            _logger.debug(f"Using hybrid search with alpha {query.alpha}")
            if vector is not None and query.query_str:
                alpha = query.alpha or 0.5

        if query.filters is not None:
            filters = _to_weaviate_filter(query.filters)
        elif "filter" in kwargs and kwargs["filter"] is not None:
            filters = kwargs["filter"]

        limit = query.similarity_top_k
        _logger.debug(f"Using limit of {query.similarity_top_k}")

        query_parameters = {
            "query": query.query_str,
            "vector": vector,
            "alpha": alpha,
            "limit": limit,
            "filters": filters,
            "return_metadata": return_metatada,
            "include_vector": True,
        }
        query_parameters.update(kwargs)  # TODO is this being tested?
        return query_parameters

    def parse_query_result(
        self, query_result: Any, query: VectorStoreQuery
    ) -> VectorStoreQueryResult:
        entries = query_result.objects

        similarity_key = "score"
        similarities = []
        nodes: List[BaseNode] = []
        node_ids = []

        for i, entry in enumerate(entries):
            if i < query.similarity_top_k:
                entry_as_dict = entry.__dict__
                similarities.append(get_node_similarity(entry_as_dict, similarity_key))
                nodes.append(to_node(entry_as_dict, text_key=self.text_key))
                node_ids.append(nodes[-1].node_id)
            else:
                break

        return VectorStoreQueryResult(
            nodes=nodes, ids=node_ids, similarities=similarities
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        collection = self._client.collections.get(self.index_name)
        query_parameters = self.get_query_parameters(query, **kwargs)

        # execute query
        try:
            query_result = collection.query.hybrid(**query_parameters)
        except weaviate.exceptions.WeaviateQueryError as e:
            raise ValueError(f"Invalid query, got errors: {e.message}")

        # parse results
        return self.parse_query_result(query_result, query)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        collection = self._aclient.collections.get(self.index_name)
        query_parameters = self.get_query_parameters(query, **kwargs)

        # execute query
        try:
            query_result = await collection.query.hybrid(**query_parameters)
        except weaviate.exceptions.WeaviateQueryError as e:
            raise ValueError(f"Invalid query, got errors: {e.message}")

        # parse results
        return self.parse_query_result(query_result, query)
