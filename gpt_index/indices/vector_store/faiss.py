"""Faiss Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from typing import Any, List, cast

import numpy as np

from gpt_index.indices.vector_store.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


class FaissVectorStore(VectorStore):
    """GPT Faiss Index.

    The GPTFaissIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Faiss index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Faiss.

    During query time, the index uses Faiss to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        faiss_index (faiss.Index): A Faiss Index object (required). Note: the index
            will be reset during index construction.
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    def __init__(
        self,
        faiss_index: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = """
            `faiss` package not found. For instructions on
            how to install `faiss` please visit
            https://github.com/facebookresearch/faiss/wiki/Installing-Faiss
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        self._faiss_index = cast(faiss.Index, faiss_index)

    @property
    def config_dict(self) -> dict:
        return {}

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add document to index."""
        new_ids = []
        for result in embedding_results:
            text_embedding = result.embedding
            text_embedding_np = np.array(text_embedding, dtype="float32")[np.newaxis, :]
            new_id = str(self._faiss_index.ntotal)
            self._faiss_index.add(text_embedding_np)
            new_ids.append(new_id)
        return new_ids

    @property
    def client(self) -> Any:
        return self._faiss_index


    @classmethod
    def load(
        cls, save_path: str
    ) -> "FaissVectorStore":
        """Load index from disk.

        This method loads the index from a JSON file stored on disk. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).
        In GPTFaissIndex, we allow user to specify an additional
        `faiss_index_save_path` to load faiss index from a file - that
        way, the user does not have to recreate the faiss index outside
        of this class.

        Args:
            save_path (str): The save_path of the file.
            faiss_index_save_path (Optional[str]): The save_path of the
                Faiss index file. If not specified, the Faiss index
                will not be saved to disk.
            **kwargs: Additional kwargs to pass to the index constructor.

        Returns:
            BaseGPTIndex: The loaded index.

        """
        import faiss

        faiss_index = faiss.read_index(save_path)
        return cls(faiss_index)

    def save(
        self,
        save_path: str,
    ) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.
        In GPTFaissIndex, we allow user to specify an additional
        `faiss_index_save_path` to save the faiss index to a file - that
        way, the user can pass in the same argument in
        `GPTFaissIndex.load_from_disk` without having to recreate
        the Faiss index outside of this class.

        Args:
            save_path (str): The save_path of the file.
            faiss_index_save_path (Optional[str]): The save_path of the
                Faiss index file. If not specified, the Faiss index
                will not be saved to disk.

        """
        import faiss

        faiss.write_index(self._faiss_index, save_path)

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not yet implemented for Faiss index.")

    def query(self, query_embedding: List[float], similarity_top_k: int) -> VectorStoreQueryResult:
        """Get nodes for response."""
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = self._faiss_index.search(
            query_embedding_np, similarity_top_k
        )
        dists = [d[0] for d in dists]
        # if empty, then return an empty response
        if len(indices) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[])

        # returned dimension is 1 x k
        node_idxs = list([str(i) for i in indices[0]])

        return VectorStoreQueryResult(similarities=dists, ids=node_idxs)