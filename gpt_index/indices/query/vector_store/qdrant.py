"""Qdrant vector store index query."""
import logging
from typing import Any, List, Optional, cast

from gpt_index.data_structs import Node, QdrantIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.vector_store.base import BaseGPTVectorStoreIndexQuery
from gpt_index.indices.utils import truncate_text


class GPTQdrantIndexQuery(BaseGPTVectorStoreIndexQuery[QdrantIndexStruct]):
    """GPTQdrantIndex query.

    An embedding-based query for GPTQdrantIndex, which queries
    an undelrying Qdrant index to retrieve top-k nodes by
    embedding similarity to the query.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        similarity_top_k (int): Number of similar nodes to retrieve.
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package

    """

    def __init__(
        self,
        index_struct: QdrantIndexStruct,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: int = 1,
        client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            **kwargs,
        )

        import_err_msg = (
            "`qdrant-client` package not found, please run `pip install qdrant-client`"
        )
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if client is None:
            raise ValueError("client cannot be None.")

        self._client = cast(qdrant_client.QdrantClient, client)

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        from qdrant_client.http.models.models import Payload

        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )

        response = self._client.search(
            collection_name=self.index_struct.get_collection_name(),
            query_vector=query_embedding,
            limit=cast(int, self.similarity_top_k),
        )

        logging.debug(f"> Top {len(response)} nodes:")

        nodes = []
        for point in response:
            payload = cast(Payload, point.payload)
            node = Node(
                ref_doc_id=payload.get("doc_id"),
                text=payload.get("text"),
            )
            nodes.append(node)

            if similarity_tracker is not None:
                similarity_tracker.add(node, point.score)

            logging.debug(
                f"> [Node {point.id}] [Similarity score: {point.score:.6}] "
                f"{truncate_text(str(payload.get('text')), 100)}"
            )

        return nodes
