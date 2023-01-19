"""Pinecone vector store index query."""
from typing import Any, Dict, List, Optional, cast

from gpt_index.data_structs.data_structs import IndexDict, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.vector_store.base import BaseGPTVectorStoreIndexQuery
from gpt_index.indices.utils import truncate_text


class GPTPineconeIndexQuery(BaseGPTVectorStoreIndexQuery[IndexDict]):
    """GPTPineconeIndex query.

    An embedding-based query for GPTPineconeIndex, which queries
    an undelrying Pinecone index to retrieve top-k nodes by
    embedding similarity to the query.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        pinecone_index (pinecone.Index): A Pinecone Index object (required)
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        similarity_top_k (int): Number of similar nodes to retrieve.

    """

    def __init__(
        self,
        index_struct: IndexDict,
        pinecone_index: Optional[Any] = None,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: Optional[int] = 1,
        pinecone_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            **kwargs,
        )
        if pinecone_index is None:
            raise ValueError("pinecone_index cannot be None.")
        # NOTE: cast to Any for now
        self._pinecone_index = cast(Any, pinecone_index)
        self._pinecone_index = pinecone_index

        self._pinecone_kwargs = pinecone_kwargs or {}

    def _get_nodes_for_response(
        self,
        query_str: str,
        verbose: bool = False,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        query_embedding = self._embed_model.get_query_embedding(query_str)

        response = self._pinecone_index.query(
            query_embedding,
            top_k=self.similarity_top_k,
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
            if similarity_tracker is not None:
                similarity_tracker.add(node, match.score)

        # print verbose output
        if verbose:
            fmt_txts = []
            for node_idx, node_similarity, node in zip(
                top_k_ids, top_k_scores, top_k_nodes
            ):
                fmt_txt = f"> [Node {node_idx}] [Similarity score: \
                    {node_similarity:.6}] {truncate_text(node.get_text(), 100)}"
                fmt_txts.append(fmt_txt)
            top_k_node_text = "\n".join(fmt_txts)
            print(f"> Top {len(top_k_nodes)} nodes:\n{top_k_node_text}")

        return top_k_nodes
