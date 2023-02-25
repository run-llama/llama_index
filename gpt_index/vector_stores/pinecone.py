"""Pinecone Vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Dict, List, Optional, cast

from gpt_index.data_structs.data_structs import Node
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


class PineconeVectorStore(VectorStore):
    """Pinecone Vector Store.

    In this vector store, embeddings and docs are stored within a
    Pinecone index.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes.

    Args:
        pinecone_index (Optional[pinecone.Index]): Pinecone index instance
        pinecone_kwargs (Optional[Dict]): kwargs to pass to Pinecone index

    """

    stores_text: bool = True

    def __init__(
        self,
        pinecone_index: Optional[Any] = None,
        pinecone_kwargs: Optional[Dict] = None,
    ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`pinecone` package not found, please run `pip install pinecone-client`"
        )
        try:
            import pinecone  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        self._pinecone_index = cast(pinecone.Index, pinecone_index)

        self._pinecone_kwargs = pinecone_kwargs or {}

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return self._pinecone_kwargs

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        ids = []
        for result in embedding_results:
            new_id = result.id
            node = result.node
            text_embedding = result.embedding

            metadata = {
                "text": node.get_text(),
                "doc_id": result.doc_id,
            }
            self._pinecone_index.upsert(
                [(new_id, text_embedding, metadata)], **self._pinecone_kwargs
            )
            ids.append(new_id)
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        # delete by filtering on the doc_id metadata
        self._pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}, **self._pinecone_kwargs
        )

    @property
    def client(self) -> Any:
        """Return Pinecone client."""
        return self._pinecone_index

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        response = self._pinecone_index.query(
            query_embedding,
            top_k=similarity_top_k,
            include_values=True,
            include_metadata=True,
            **self._pinecone_kwargs,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for match in response.matches:
            text = match.metadata["text"]
            node = Node(text=text, extra_info=match.metadata)
            top_k_ids.append(match.id)
            top_k_nodes.append(node)
            top_k_scores.append(match.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
