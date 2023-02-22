"""Weaviate Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from typing import Any, List, Optional, cast

from gpt_index.readers.weaviate.data_structs import WeaviateNode
from gpt_index.readers.weaviate.utils import get_default_class_prefix
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


class WeaviateVectorStore(VectorStore):
    """GPT Weaviate Index.

    The GPTWeaviateIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Weaviate index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Weaviate.

    During query time, the index uses Weaviate to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

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
            raise ValueError(import_err_msg)

        self._client = cast(Client, weaviate_client)
        self._class_prefix = class_prefix or get_default_class_prefix()
        # try to create schema
        WeaviateNode.create_schema(self._client, self._class_prefix)

    @property
    def client(self) -> Any:
        return self._client

    @property
    def config_dict(self) -> dict:
        return {"class_prefix": self._class_prefix}

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> None:
        """Add document to index."""
        for result in embedding_results:
            node = result.node
            embedding = result.embedding
            # TODO: always store embedding in node
            node.embedding = embedding
            WeaviateNode.from_gpt_index(self._client, node, self._class_prefix)

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        WeaviateNode.delete_document(self._client, doc_id, self._class_prefix)

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        nodes = WeaviateNode.to_gpt_index_list(
            self.client,
            self._class_prefix,
            vector=query_embedding,
            object_limit=similarity_top_k,
        )
        nodes = nodes[:similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]

        return VectorStoreQueryResult(nodes=nodes, ids=node_idxs)
