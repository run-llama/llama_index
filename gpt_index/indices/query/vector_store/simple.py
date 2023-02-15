"""Default query for GPTSimpleVectorIndex."""
import logging
from typing import List, Optional

from gpt_index.data_structs.data_structs import Node, SimpleIndexDict
from gpt_index.indices.query.embedding_utils import (
    SimilarityTracker,
    get_top_k_embeddings,
)
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.vector_store.base import BaseGPTVectorStoreIndexQuery
from gpt_index.indices.utils import truncate_text


class GPTSimpleVectorIndexQuery(BaseGPTVectorStoreIndexQuery[SimpleIndexDict]):
    """GPTSimpleVectorIndex query.

    An embedding-based query for GPTSimpleVectorIndex, which queries
    an underlying dict-based embedding store to retrieve top-k nodes by
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

    """

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        # TODO: consolidate with get_query_text_embedding_similarities
        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )
        items = self._index_struct.embedding_dict.items()
        node_ids = [t[0] for t in items]
        embeddings = [t[1] for t in items]

        top_similarities, top_ids = get_top_k_embeddings(
            self._embed_model,
            query_embedding,
            embeddings,
            similarity_top_k=self.similarity_top_k,
            embedding_ids=node_ids,
        )
        top_k_nodes = self._index_struct.get_nodes(top_ids)
        if similarity_tracker is not None:
            for node, similarity in zip(top_k_nodes, top_similarities):
                similarity_tracker.add(node, similarity)

        if logging.getLogger(__name__).getEffectiveLevel() == logging.DEBUG:
            fmt_txts = []
            for node_idx, node_similarity, node in zip(
                top_ids, top_similarities, top_k_nodes
            ):
                fmt_txt = f"> [Node {node_idx}] [Similarity score: \
                    {node_similarity:.6}] {truncate_text(node.get_text(), 100)}"
                fmt_txts.append(fmt_txt)
            top_k_node_text = "\n".join(fmt_txts)
            logging.debug(f"> Top {len(top_k_nodes)} nodes:\n{top_k_node_text}")

        return top_k_nodes
