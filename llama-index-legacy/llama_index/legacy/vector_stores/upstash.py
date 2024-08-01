"""
Upstash vector store index.

An index that is built with Upstash Vector.

https://upstash.com/docs/vector/overall/getstarted
"""

import logging
from typing import Any, List

from llama_index.legacy.schema import BaseNode
from llama_index.legacy.utils import iter_batch
from llama_index.legacy.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 128


class UpstashVectorStore(VectorStore):
    """
    Upstash Vector Store.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    @classmethod
    def class_name(cls) -> str:
        return "UpstashVectorStore"

    @property
    def client(self) -> Any:
        """Return the Upstash client."""
        return self._index

    def __init__(
        self, url: str, token: str, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """
        Create a UpstashVectorStore. The index can be created using the Upstash console.

        Args:
            url (String): URL of the Upstash Vector instance, found in the Upstash console.
            token (String): Token for the Upstash Vector Index, found in the Upstash console.
            batch_size (Optional[int]): Batch size for adding nodes to the vector store.

        Raises:
            ImportError: If the upstash-vector python package is not installed.
        """
        self.batch_size = batch_size

        try:
            from upstash_vector import Index
        except ImportError:
            raise ImportError(
                "Could not import upstash_vector.Index, Please install it with `pip install upstash-vector`"
            )

        self._index = Index(url=url, token=token)

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes: List of nodes to add to the vector store.
            add_kwargs: Additional arguments to pass to the add method.

        Returns:
            List of ids of the added nodes.
        """
        ids = []
        vectors = []
        for node_batch in iter_batch(nodes, self.batch_size):
            for node in node_batch:
                metadata_dict = node_to_metadata_dict(node)
                ids.append(node.node_id)
                vectors.append((node.node_id, node.embedding, metadata_dict))

            self.client.upsert(vectors=vectors)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete node from the vector store.

        Args:
            ref_doc_id: Reference doc id of the node to delete.
            delete_kwargs: Additional arguments to pass to the delete method.
        """
        raise NotImplementedError(
            "Delete is not currently supported, but will be in the future."
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store.

        Args:
            query: Query to run against the vector store.
            kwargs: Additional arguments to pass to the query method.

        Returns:
            Query result.
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Query mode {query.mode} not supported")

        if query.filters:
            raise ValueError("Metadata filtering not supported")

        res = self.client.query(
            vector=query.query_embedding,
            top_k=query.similarity_top_k,
            include_vectors=True,
            include_metadata=True,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for vector in res:
            node = metadata_dict_to_node(vector.metadata)
            node.embedding = vector.vector
            top_k_nodes.append(node)
            top_k_ids.append(vector.id)
            top_k_scores.append(vector.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
