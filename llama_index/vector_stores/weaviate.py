"""Weaviate Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.schema import BaseNode
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import DEFAULT_TEXT_KEY
from llama_index.vector_stores.weaviate_utils import (
    add_node,
    class_schema_exists,
    create_default_schema,
    get_all_properties,
    get_node_similarity,
    parse_get_response,
    to_node,
)

logger = logging.getLogger(__name__)

import_err_msg = (
    "`weaviate` package not found, please run `pip install weaviate-client`"
)


def _transform_weaviate_filter_condition(condition: str) -> str:
    """Translate standard metadata filter op to Chroma specific spec."""
    if condition == "and":
        return "And"
    elif condition == "or":
        return "Or"
    else:
        raise ValueError(f"Filter condition {condition} not supported")


def _transform_weaviate_filter_operator(operator: str) -> str:
    """Translate standard metadata filter operator to Chroma specific spec."""
    if operator == "!=":
        return "NotEqual"
    elif operator == "==":
        return "Equal"
    elif operator == ">":
        return "GreaterThan"
    elif operator == "<":
        return "LessThan"
    elif operator == ">=":
        return "GreaterThanEqual"
    elif operator == "<=":
        return "LessThanEqual"
    else:
        raise ValueError(f"Filter operator {operator} not supported")


def _to_weaviate_filter(standard_filters: MetadataFilters) -> Dict[str, Any]:
    filters_list = []
    condition = standard_filters.condition or "and"
    condition = _transform_weaviate_filter_condition(condition)

    if standard_filters.filters:
        for filter in standard_filters.filters:
            value_type = "valueText"
            if isinstance(filter.value, float):
                value_type = "valueNumber"
            elif isinstance(filter.value, int):
                value_type = "valueNumber"
            elif isinstance(filter.value, str) and filter.value.isnumeric():
                filter.value = float(filter.value)
                value_type = "valueNumber"
            filters_list.append(
                {
                    "path": filter.key,
                    "operator": _transform_weaviate_filter_operator(filter.operator),
                    value_type: filter.value,
                }
            )
    else:
        return {}

    if len(filters_list) == 1:
        # If there is only one filter, return it directly
        return filters_list[0]

    return {"operands": filters_list, "operator": condition}


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

    """

    stores_text: bool = True

    index_name: str
    url: Optional[str]
    text_key: str
    auth_config: Dict[str, Any] = Field(default_factory=dict)
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _client = PrivateAttr()

    def __init__(
        self,
        weaviate_client: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        index_name: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        auth_config: Optional[Any] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        try:
            import weaviate  # noqa
            from weaviate import AuthApiKey, Client
        except ImportError:
            raise ImportError(import_err_msg)

        if weaviate_client is None:
            if isinstance(auth_config, dict):
                auth_config = AuthApiKey(**auth_config)

            client_kwargs = client_kwargs or {}
            self._client = Client(
                url=url, auth_client_secret=auth_config, **client_kwargs
            )
        else:
            self._client = cast(Client, weaviate_client)

        # validate class prefix starts with a capital letter
        if class_prefix is not None:
            logger.warning("class_prefix is deprecated, please use index_name")
            # legacy, kept for backward compatibility
            index_name = f"{class_prefix}_Node"

        index_name = index_name or f"LlamaIndex_{uuid4().hex}"
        if not index_name[0].isupper():
            raise ValueError(
                "Index name must start with a capital letter, e.g. 'LlamaIndex'"
            )

        # create default schema if does not exist
        if not class_schema_exists(self._client, index_name):
            create_default_schema(self._client, index_name)

        super().__init__(
            url=url,
            index_name=index_name,
            text_key=text_key,
            auth_config=auth_config.__dict__ if auth_config else {},
            client_kwargs=client_kwargs or {},
        )

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
        try:
            import weaviate  # noqa
            from weaviate import AuthApiKey, Client  # noqa
        except ImportError:
            raise ImportError(import_err_msg)

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

        with self._client.batch as batch:
            for node in nodes:
                add_node(
                    self._client,
                    node,
                    self.index_name,
                    batch=batch,
                    text_key=self.text_key,
                )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        where_filter = {
            "path": ["ref_doc_id"],
            "operator": "Equal",
            "valueText": ref_doc_id,
        }
        if "filter" in delete_kwargs and delete_kwargs["filter"] is not None:
            where_filter = {
                "operator": "And",
                "operands": [where_filter, delete_kwargs["filter"]],  # type: ignore
            }

        query = (
            self._client.query.get(self.index_name)
            .with_additional(["id"])
            .with_where(where_filter)
            .with_limit(10000)  # 10,000 is the max weaviate can fetch
        )

        query_result = query.do()
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[self.index_name]
        for entry in entries:
            self._client.data_object.delete(entry["_additional"]["id"], self.index_name)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        all_properties = get_all_properties(self._client, self.index_name)

        # build query
        query_builder = self._client.query.get(self.index_name, all_properties)

        # list of documents to constrain search
        if query.doc_ids:
            filter_with_doc_ids = {
                "operator": "Or",
                "operands": [
                    {"path": ["doc_id"], "operator": "Equal", "valueText": doc_id}
                    for doc_id in query.doc_ids
                ],
            }
            query_builder = query_builder.with_where(filter_with_doc_ids)

        if query.node_ids:
            filter_with_node_ids = {
                "operator": "Or",
                "operands": [
                    {"path": ["id"], "operator": "Equal", "valueText": node_id}
                    for node_id in query.node_ids
                ],
            }
            query_builder = query_builder.with_where(filter_with_node_ids)

        query_builder = query_builder.with_additional(
            ["id", "vector", "distance", "score"]
        )

        vector = query.query_embedding
        similarity_key = "distance"
        if query.mode == VectorStoreQueryMode.DEFAULT:
            logger.debug("Using vector search")
            if vector is not None:
                query_builder = query_builder.with_near_vector(
                    {
                        "vector": vector,
                    }
                )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            logger.debug(f"Using hybrid search with alpha {query.alpha}")
            similarity_key = "score"
            if vector is not None and query.query_str:
                query_builder = query_builder.with_hybrid(
                    query=query.query_str,
                    alpha=query.alpha,
                    vector=vector,
                )

        if query.filters is not None:
            filter = _to_weaviate_filter(query.filters)
            query_builder = query_builder.with_where(filter)
        elif "filter" in kwargs and kwargs["filter"] is not None:
            query_builder = query_builder.with_where(kwargs["filter"])

        query_builder = query_builder.with_limit(query.similarity_top_k)
        logger.debug(f"Using limit of {query.similarity_top_k}")

        # execute query
        query_result = query_builder.do()

        # parse results
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[self.index_name]

        similarities = []
        nodes: List[BaseNode] = []
        node_ids = []

        for i, entry in enumerate(entries):
            if i < query.similarity_top_k:
                similarities.append(get_node_similarity(entry, similarity_key))
                nodes.append(to_node(entry, text_key=self.text_key))
                node_ids.append(nodes[-1].node_id)
            else:
                break

        return VectorStoreQueryResult(
            nodes=nodes, ids=node_ids, similarities=similarities
        )
