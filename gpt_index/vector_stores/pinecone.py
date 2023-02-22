"""Pinecone Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from typing import Any, Dict, List, Optional, cast

from gpt_index.data_structs.data_structs import Node
from gpt_index.utils import get_new_id
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


class PineconeVectorStore(VectorStore):
    """GPT Pinecone Index.

    The GPTPineconeIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Pinecone index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Pinecone.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        chunk_size_limit (int): Maximum number of tokens per chunk. NOTE:
            in Pinecone the default is 2048 due to metadata size restrictions.
    """

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
        return self._pinecone_kwargs

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> None:
        """Add document to index."""
        for result in embedding_results:
            new_id = result.id
            node = result.node
            text_embedding = result.embedding

            # assign a new_id if current_id conflicts with existing ids
            while True:
                fetch_result = self._pinecone_index.fetch(
                    [new_id], **self._pinecone_kwargs
                )
                if len(fetch_result["vectors"]) == 0:
                    break
                new_id = get_new_id(set())
            metadata = {
                "text": node.get_text(),
                "doc_id": result.doc_id,
            }
            self._pinecone_index.upsert(
                [(new_id, text_embedding, metadata)], **self._pinecone_kwargs
            )

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        # delete by filtering on the doc_id metadata
        self._pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}, **self._pinecone_kwargs
        )

    @property
    def client(self) -> Any:
        return self._pinecone_index

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
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
