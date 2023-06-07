"""Simple vector store index."""

import json
import logging
import os
import fsspec
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

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
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

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
        text_id_to_ref_doc_id (Optional[dict]): dict mapping text_ids to ref_doc_ids.

    """

    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    text_id_to_ref_doc_id: Dict[str, str] = field(default_factory=dict)


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
        fs: Optional[fsspec.AbstractFileSystem] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._data = data or SimpleVectorStoreData()
        self._fs = fs or fsspec.filesystem("file")

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleVectorStore":
        """Load from persist dir."""
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        return cls.from_persist_path(persist_path, fs=fs)

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
            self._data.text_id_to_ref_doc_id[result.id] = result.ref_doc_id
        return [result.id for result in embedding_results]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        text_ids_to_delete = set()
        for text_id, ref_doc_id_ in self._data.text_id_to_ref_doc_id.items():
            if ref_doc_id == ref_doc_id_:
                text_ids_to_delete.add(text_id)

        for text_id in text_ids_to_delete:
            del self._data.embedding_dict[text_id]
            del self._data.text_id_to_ref_doc_id[text_id]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        if query.filters is not None:
            raise ValueError(
                "Metadata filters not implemented for SimpleVectorStore yet."
            )

        # TODO: consolidate with get_query_text_embedding_similarities
        items = self._data.embedding_dict.items()

        if query.doc_ids:
            available_ids = set(query.doc_ids)

            node_ids = [t[0] for t in items if t[0] in available_ids]
            embeddings = [t[1] for t in items if t[0] in available_ids]
        else:
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

    def persist(
        self,
        persist_path: str = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME),
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the SimpleVectorStore to a directory."""
        fs = fs or self._fs
        dirpath = os.path.dirname(persist_path)
        if not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "w") as f:
            json.dump(self._data.to_dict(), f)

    @classmethod
    def from_persist_path(
        cls, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> "SimpleVectorStore":
        """Create a SimpleKVStore from a persist directory."""
        fs = fs or fsspec.filesystem("file")
        if not fs.exists(persist_path):
            raise ValueError(
                f"No existing {__name__} found at {persist_path}, skipping load."
            )

        logger.debug(f"Loading {__name__} from {persist_path}.")
        with fs.open(persist_path, "rb") as f:
            data_dict = json.load(f)
            data = SimpleVectorStoreData.from_dict(data_dict)
        return cls(data)

    @classmethod
    def from_dict(cls, save_dict: dict) -> "SimpleVectorStore":
        data = SimpleVectorStoreData.from_dict(save_dict)
        return cls(data)

    def to_dict(self) -> dict:
        return self._data.to_dict()
