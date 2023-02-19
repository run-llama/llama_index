"""Simple vector store index."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from gpt_index.indices.query.embedding_utils import get_top_k_embeddings
from gpt_index.indices.vector_store.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


@dataclass
class SimpleVectorStoreData(DataClassJsonMixin):
    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    text_id_to_doc_id: Dict[str, str] = field(default_factory=dict)


class SimpleVectorStore(VectorStore):
    """GPT Simple Vector Index.

    The GPTSimpleVectorIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a simple dictionary.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within the dict.

    During query time, the index uses the dict to query for the top
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
        simple_vector_store_data_dict: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if simple_vector_store_data_dict is None:
            self._data = SimpleVectorStoreData()
        else:
            self._data = SimpleVectorStoreData.from_dict(simple_vector_store_data_dict)

    @property
    def client(self) -> None:
        return None
    
    @property
    def config_dict(self) -> dict:
        return {
            "simple_vector_store_data_dict": self._data.to_dict(),
        }

    def get(self, text_id: str) -> List[float]:
        return self._data.embedding_dict[text_id]

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add document to index."""
        for result in embedding_results:
            text_id = result.id
            self._data.embedding_dict[text_id] = result.embedding
            self._data.text_id_to_doc_id[text_id] = result.doc_id
        return [result.id for result in embedding_results]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        text_ids_to_delete = set()
        for text_id, doc_id_ in self._data.text_id_to_doc_id.items():
            if doc_id == doc_id_:
                text_ids_to_delete.add(text_id)

        for text_id in text_ids_to_delete:
            del self._data.embedding_dict[text_id]
            del self._data.text_id_to_doc_id[text_id]

    def query(self, query_embedding: List[float], similarity_top_k: int) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # TODO: consolidate with get_query_text_embedding_similarities
        items = self._data.embedding_dict.items()
        node_ids = [t[0] for t in items]
        embeddings = [t[1] for t in items]

        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=similarity_top_k,
            embedding_ids=node_ids,
        )

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
