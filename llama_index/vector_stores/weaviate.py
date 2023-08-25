"""Weaviate Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
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


def _to_weaviate_filter(standard_filters: MetadataFilters) -> Dict[str, Any]:
    if len(standard_filters.filters) == 1:
        return {
            "path": standard_filters.filters[0].key,
            "operator": "Equal",
            "valueText": standard_filters.filters[0].value,
        }
    else:
        operands = []
        for filter in standard_filters.filters:
            operands.append(
                {"path": filter.key, "operator": "Equal", "valueText": filter.value}
            )
        return {"operands": operands, "operator": "And"}


class WeaviateVectorStore(VectorStore):
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

    def __init__(
        self,
        weaviate_client: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        index_name: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`weaviate` package not found, please run `pip install weaviate-client`"
        )
        try:
            import weaviate  # noqa: F401
            from weaviate import Client  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if weaviate_client is None:
            raise ValueError("Missing Weaviate client!")

        self._client = cast(Client, weaviate_client)
        # validate class prefix starts with a capital letter
        if class_prefix is not None:
            logger.warning("class_prefix is deprecated, please use index_name")
            # legacy, kept for backward compatibility
            index_name = f"{class_prefix}_Node"

        self._index_name = index_name or f"LlamaIndex_{uuid4().hex}"
        if not self._index_name[0].isupper():
            raise ValueError(
                "Index name must start with a capital letter, e.g. 'LlamaIndex'"
            )
        self._text_key = text_key

        # create default schema if does not exist
        if not class_schema_exists(self._client, self._index_name):
            create_default_schema(self._client, self._index_name)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        for result in embedding_results:
            node = result.node
            embedding = result.embedding
            # TODO: always store embedding in node
            node.embedding = embedding

        nodes = [r.node for r in embedding_results]
        ids = [r.id for r in embedding_results]

        with self._client.batch as batch:
            for node in nodes:
                add_node(
                    self._client,
                    node,
                    self._index_name,
                    batch=batch,
                    text_key=self._text_key,
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
            "valueString": ref_doc_id,
        }
        query = (
            self._client.query.get(self._index_name)
            .with_additional(["id"])
            .with_where(where_filter)
        )

        query_result = query.do()
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[self._index_name]
        for entry in entries:
            self._client.data_object.delete(
                entry["_additional"]["id"], self._index_name
            )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""

        all_properties = get_all_properties(self._client, self._index_name)

        # build query
        query_builder = self._client.query.get(self._index_name, all_properties)

        # list of documents to constrain search
        if query.doc_ids:
            filter_with_doc_ids = {
                "operator": "Or",
                "operands": [
                    {"path": ["doc_id"], "operator": "Equal", "valueString": doc_id}
                    for doc_id in query.doc_ids
                ],
            }
            query_builder = query_builder.with_where(filter_with_doc_ids)

        if query.node_ids:
            filter_with_node_ids = {
                "operator": "Or",
                "operands": [
                    {"path": ["id"], "operator": "Equal", "valueString": node_id}
                    for node_id in query.node_ids
                ],
            }
            query_builder = query_builder.with_where(filter_with_node_ids)

        query_builder = query_builder.with_additional(["id", "vector", "distance"])

        vector = query.query_embedding
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
            query_builder = query_builder.with_hybrid(
                query=query.query_str,
                alpha=query.alpha,
                vector=vector,
            )

        if query.filters is not None and len(query.filters.filters) > 0:
            filter = _to_weaviate_filter(query.filters)
            query_builder = query_builder.with_where(filter)
        else:
            filter = kwargs.pop("filter", {})

        query_builder = query_builder.with_limit(query.similarity_top_k)
        logger.debug(f"Using limit of {query.similarity_top_k}")

        # execute query
        query_result = query_builder.do()

        # parse results
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[self._index_name]

        similarities = [get_node_similarity(entry) for entry in entries]
        nodes = [to_node(entry, text_key=self._text_key) for entry in entries]

        nodes = nodes[: query.similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]

        return VectorStoreQueryResult(
            nodes=nodes, ids=node_idxs, similarities=similarities
        )
