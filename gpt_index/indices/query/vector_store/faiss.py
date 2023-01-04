"""Default query for GPTFaissIndex."""
from typing import Any, List, Optional, cast

import numpy as np

from gpt_index.data_structs.data_structs import IndexDict, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.vector_store.base import BaseGPTVectorStoreIndexQuery


class GPTFaissIndexQuery(BaseGPTVectorStoreIndexQuery[IndexDict]):
    """GPTFaissIndex query.

    An embedding-based query for GPTFaissIndex, which queries
    an undelrying Faiss index to retrieve top-k nodes by
    embedding similarity to the query.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        faiss_index (faiss.Index): A Faiss Index object (required)
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        similarity_top_k (int): Number of similar nodes to retrieve.

    """

    def __init__(
        self,
        index_struct: IndexDict,
        faiss_index: Optional[Any] = None,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            **kwargs,
        )
        if faiss_index is None:
            raise ValueError("faiss_index cannot be None.")
        # NOTE: cast to Any for now
        self._faiss_index = cast(Any, faiss_index)
        self._faiss_index = faiss_index

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        query_embedding_np = np.array(query_embedding)[np.newaxis, :]
        _, indices = self._faiss_index.search(query_embedding_np, self.similarity_top_k)
        # if empty, then return an empty response
        if len(indices) == 0:
            return []

        # returned dimension is 1 x k
        node_idxs = list([str(i) for i in indices[0]])
        top_k_nodes = self._index_struct.get_nodes(node_idxs)

        return top_k_nodes
