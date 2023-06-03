"""Weaviate Vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, List, Optional, cast

from llama_index.readers.weaviate.client import (
    add_nodes,
    create_schema,
    delete_document,
    weaviate_query,
)
from llama_index.readers.weaviate.utils import get_default_class_prefix
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


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
        if class_prefix is not None and not class_prefix[0].isupper():
            raise ValueError(
                "Class prefix must start with a capital letter, e.g. 'Gpt'"
            )
        self._class_prefix = class_prefix or get_default_class_prefix()
        # try to create schema
        create_schema(self._client, self._class_prefix)

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

        add_nodes(self._client, [r.node for r in embedding_results], self._class_prefix)
        return [result.id for result in embedding_results]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        delete_document(self._client, ref_doc_id, self._class_prefix)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Weaviate yet.")

        nodes = weaviate_query(
            self._client,
            self._class_prefix,
            query,
        )
        nodes = nodes[: query.similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]

        return VectorStoreQueryResult(nodes=nodes, ids=node_idxs)
