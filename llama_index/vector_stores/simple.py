"""Simple vector store index."""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, cast, Optional

from dataclasses_json import DataClassJsonMixin

from llama_index.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
)
from llama_index.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

import logging

logger = logging.getLogger(__name__)

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}


@dataclass
class SimpleVectorStoreData(DataClassJsonMixin):
    """Simple Vector Store Data container.

    Args:
        embedding_dict (Optional[dict]): dict mapping doc_ids to embeddings.
        text_id_to_doc_id (Optional[dict]): dict mapping text_ids to doc_ids.

    """

    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    text_id_to_doc_id: Dict[str, str] = field(default_factory=dict)


class SimpleVectorStore(VectorStore):
    """Simple Vector Store.

    In this vector store, embeddings are stored within a simple, in-memory dictionary.

    Args:
        simple_vector_store_data_dict (Optional[dict]): data dict
            containing the embeddings and doc_ids. See SimpleVectorStoreData
            for more details.
    """

    stores_text: bool = False

    def __init__(
        self,
        data: Optional[SimpleVectorStoreData] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._data = data or SimpleVectorStoreData()

    @classmethod
    def from_persist_dir(
        cls, persist_dir: str = DEFAULT_PERSIST_DIR
    ) -> "SimpleVectorStore":
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        return cls.from_persist_path(persist_path)

    @property
    def client(self) -> None:
        """Get client."""
        return None

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self._data.embedding_dict[text_id]

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding_results to index."""
        for result in embedding_results:
            self._data.embedding_dict[result.id] = result.embedding
            self._data.text_id_to_doc_id[result.id] = result.ref_doc_id
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

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # TODO: consolidate with get_query_text_embedding_similarities
        items = self._data.embedding_dict.items()
        node_ids = [t[0] for t in items]
        embeddings = [t[1] for t in items]

        query_embedding = cast(List[float], query.query_embedding)

        if query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)

    def persist(self, persist_path: str) -> None:
        """Persist the SimpleVectorStore to a directory."""
        dirpath = os.path.dirname(persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(persist_path, "w+") as f:
            json.dump(self._data.to_dict(), f)

    @classmethod
    def from_persist_path(cls, persist_path: str) -> "SimpleVectorStore":
        """Create a SimpleKVStore from a persist directory."""
        if not os.path.exists(persist_path):
            raise ValueError(
                f"No existing {__name__} found at {persist_path}, skipping load."
            )

        logger.debug(f"Loading {__name__} from {persist_path}.")
        with open(persist_path, "r+") as f:
            data_dict = json.load(f)
            data = SimpleVectorStoreData.from_dict(data_dict)
        return cls(data)

    @classmethod
    def from_dict(cls, save_dict: dict) -> "SimpleVectorStore":
        data = SimpleVectorStoreData.from_dict(save_dict)
        return cls(data)

    def to_dict(self) -> dict:
        return self._data.to_dict()
