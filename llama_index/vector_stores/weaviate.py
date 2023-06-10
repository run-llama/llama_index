"""Weaviate Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from typing import Any, List, Optional, cast
from uuid import uuid4

from llama_index.vector_stores.types import (NodeWithEmbedding, VectorStore,
                                             VectorStoreQuery,
                                             VectorStoreQueryMode,
                                             VectorStoreQueryResult)
from llama_index.vector_stores.utils import DEFAULT_TEXT_KEY
from llama_index.vector_stores.weaviate_utils import (NODE_SCHEMA, add_node,
                                                      class_schema_exists,
                                                      create_default_schema,
                                                      parse_get_response,
                                                      to_node)

logger = logging.getLogger(__name__)


class WeaviateVectorStore(VectorStore):
    """Weaviate vector store.

    In this vector store, embeddings and docs are stored within a
    Weaviate collection.

    During query time, the index uses Weaviate to query for the top
    k most similar nodes.

    Args:
        weaviate_client (weaviate.Client): WeaviateClient
            instance from `weaviate-client` package
        class_prefix (Optional[str]): prefix for Weaviate classes

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
            logger.warning("class_prefix is deprecated, please use class_name")
            if not class_prefix[0].isupper():
                raise ValueError(
                    "Class prefix must start with a capital letter, e.g. 'Gpt'"
                )
            # legacy, kept for backward compatibility
            index_name = f"{class_prefix}_Node"

        self._index_name = index_name or f"LlamaIndex_{uuid4().hex}"
        self._text_key = text_key

        # create default schema if does not exist
        if class_schema_exists(self._client, self._index_name):
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
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Weaviate yet.")

        prop_names = [p["name"] for p in NODE_SCHEMA]
        vector = query.query_embedding

        # build query
        query = self._client.query.get(self._index_name, prop_names).with_additional(
            ["id", "vector"]
        )
        if query.mode == VectorStoreQueryMode.DEFAULT:
            logger.debug("Using vector search")
            if vector is not None:
                query = query.with_near_vector(
                    {
                        "vector": vector,
                    }
                )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            logger.debug(f"Using hybrid search with alpha {query.alpha}")
            query = query.with_hybrid(
                query=query.query_str,
                alpha=query.alpha,
                vector=vector,
            )
        query = query.with_limit(query.similarity_top_k)
        logger.debug(f"Using limit of {query.similarity_top_k}")

        # execute query
        query_result = query.do()

        # parse results
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[self._index_name]
        nodes = [to_node(entry, text_key=self._text_key) for entry in entries]

        nodes = nodes[: query.similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]

        return VectorStoreQueryResult(nodes=nodes, ids=node_idxs)
