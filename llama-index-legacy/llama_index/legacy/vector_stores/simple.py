"""Simple vector store index."""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

import fsspec
from dataclasses_json import DataClassJsonMixin

from llama_index.legacy.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
from llama_index.legacy.schema import BaseNode
from llama_index.legacy.utils import concat_dirs
from llama_index.legacy.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import node_to_metadata_dict

logger = logging.getLogger(__name__)

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}

MMR_MODE = VectorStoreQueryMode.MMR

NAMESPACE_SEP = "__"
DEFAULT_VECTOR_STORE = "default"


def _build_metadata_filter_fn(
    metadata_lookup_fn: Callable[[str], Mapping[str, Any]],
    metadata_filters: Optional[MetadataFilters] = None,
) -> Callable[[str], bool]:
    """Build metadata filter function."""
    filter_list = metadata_filters.legacy_filters() if metadata_filters else []
    if not filter_list:
        return lambda _: True

    def filter_fn(node_id: str) -> bool:
        metadata = metadata_lookup_fn(node_id)
        for filter_ in filter_list:
            metadata_value = metadata.get(filter_.key, None)
            if metadata_value is None:
                return False
            elif isinstance(metadata_value, list):
                if filter_.value not in metadata_value:
                    return False
            elif isinstance(metadata_value, (int, float, str, bool)):
                if metadata_value != filter_.value:
                    return False
        return True

    return filter_fn


@dataclass
class SimpleVectorStoreData(DataClassJsonMixin):
    """Simple Vector Store Data container.

    Args:
        embedding_dict (Optional[dict]): dict mapping node_ids to embeddings.
        text_id_to_ref_doc_id (Optional[dict]):
            dict mapping text_ids/node_ids to ref_doc_ids.

    """

    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    text_id_to_ref_doc_id: Dict[str, str] = field(default_factory=dict)
    metadata_dict: Dict[str, Any] = field(default_factory=dict)


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
        namespace: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleVectorStore":
        """Load from persist dir."""
        if namespace:
            persist_fname = f"{namespace}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
        else:
            persist_fname = DEFAULT_PERSIST_FNAME

        if fs is not None:
            persist_path = concat_dirs(persist_dir, persist_fname)
        else:
            persist_path = os.path.join(persist_dir, persist_fname)
        return cls.from_persist_path(persist_path, fs=fs)

    @classmethod
    def from_namespaced_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> Dict[str, VectorStore]:
        """Load from namespaced persist dir."""
        listing_fn = os.listdir if fs is None else fs.listdir

        vector_stores: Dict[str, VectorStore] = {}

        try:
            for fname in listing_fn(persist_dir):
                if fname.endswith(DEFAULT_PERSIST_FNAME):
                    namespace = fname.split(NAMESPACE_SEP)[0]

                    # handle backwards compatibility with stores that were persisted
                    if namespace == DEFAULT_PERSIST_FNAME:
                        vector_stores[DEFAULT_VECTOR_STORE] = cls.from_persist_dir(
                            persist_dir=persist_dir, fs=fs
                        )
                    else:
                        vector_stores[namespace] = cls.from_persist_dir(
                            persist_dir=persist_dir, namespace=namespace, fs=fs
                        )
        except Exception:
            # failed to listdir, so assume there is only one store
            try:
                vector_stores[DEFAULT_VECTOR_STORE] = cls.from_persist_dir(
                    persist_dir=persist_dir, fs=fs, namespace=DEFAULT_VECTOR_STORE
                )
            except Exception:
                # no namespace backwards compat
                vector_stores[DEFAULT_VECTOR_STORE] = cls.from_persist_dir(
                    persist_dir=persist_dir, fs=fs
                )

        return vector_stores

    @property
    def client(self) -> None:
        """Get client."""
        return

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self._data.embedding_dict[text_id]

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self._data.embedding_dict[node.node_id] = node.get_embedding()
            self._data.text_id_to_ref_doc_id[node.node_id] = node.ref_doc_id or "None"

            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=False
            )
            metadata.pop("_node_content", None)
            self._data.metadata_dict[node.node_id] = metadata
        return [node.node_id for node in nodes]

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
            # Handle metadata_dict not being present in stores that were persisted
            # without metadata, or, not being present for nodes stored
            # prior to metadata functionality.
            if self._data.metadata_dict is not None:
                self._data.metadata_dict.pop(text_id, None)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # Prevent metadata filtering on stores that were persisted without metadata.
        if (
            query.filters is not None
            and self._data.embedding_dict
            and not self._data.metadata_dict
        ):
            raise ValueError(
                "Cannot filter stores that were persisted without metadata. "
                "Please rebuild the store with metadata to enable filtering."
            )
        # Prefilter nodes based on the query filter and node ID restrictions.
        query_filter_fn = _build_metadata_filter_fn(
            lambda node_id: self._data.metadata_dict[node_id], query.filters
        )

        if query.node_ids is not None:
            available_ids = set(query.node_ids)

            def node_filter_fn(node_id: str) -> bool:
                return node_id in available_ids

        else:

            def node_filter_fn(node_id: str) -> bool:
                return True

        node_ids = []
        embeddings = []
        # TODO: consolidate with get_query_text_embedding_similarities
        for node_id, embedding in self._data.embedding_dict.items():
            if node_filter_fn(node_id) and query_filter_fn(node_id):
                node_ids.append(node_id)
                embeddings.append(embedding)

        query_embedding = cast(List[float], query.query_embedding)

        if query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        elif query.mode == MMR_MODE:
            mmr_threshold = kwargs.get("mmr_threshold", None)
            top_similarities, top_ids = get_top_k_mmr_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
                mmr_threshold=mmr_threshold,
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
