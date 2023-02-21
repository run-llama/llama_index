"""Chroma vector store index query."""
import logging
import math
from typing import Any, List, Optional, cast

from gpt_index.data_structs.data_structs import ChromaIndexStruct, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.vector_store.base import BaseGPTVectorStoreIndexQuery
from gpt_index.indices.utils import truncate_text


class GPTChromaIndexQuery(BaseGPTVectorStoreIndexQuery[ChromaIndexStruct]):
    """GPTChromaIndex query.

    An embedding-based query for GPTChromaIndex, which queries
    an undelrying Chroma index to retrieve top-k nodes by
    embedding similarity to the query.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        chroma_collection (chromadb.api.models.Collection.Collection): A Chroma
            Collection object (required)
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        similarity_top_k (int): Number of similar nodes to retrieve.

    """

    def __init__(
        self,
        index_struct: ChromaIndexStruct,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: int = 1,
        chroma_collection: Optional[Any] = None,
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
            "`chromadb` package not found, please run `pip install chromadb`"
        )
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if chroma_collection is None:
            raise ValueError("chroma_collection cannot be None.")

        from chromadb.api.models.Collection import Collection

        self._collection = cast(Collection, chroma_collection)

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )

        results = self._collection.query(
            query_embeddings=query_embedding, n_results=self.similarity_top_k
        )

        logging.debug(f"> Top {len(results['documents'])} nodes:")
        nodes = []
        for result in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["distances"],
        ):
            node = Node(
                ref_doc_id=result[0][0],
                text=result[1][0],
                extra_info=result[2][0],
            )
            nodes.append(node)

            similarity_score = 1.0 - math.exp(-result[3][0])

            if similarity_tracker is not None:
                similarity_tracker.add(node, similarity_score)

            logging.debug(
                f"> [Node {result[0][0]}] [Similarity score: {similarity_score}] "
                f"{truncate_text(str(result[1][0]), 100)}"
            )

        return nodes
