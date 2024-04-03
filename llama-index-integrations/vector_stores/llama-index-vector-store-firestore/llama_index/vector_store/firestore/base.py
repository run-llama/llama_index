"""Firestore vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, List, Optional

import more_itertools
from google.cloud.firestore import Client
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.document import DocumentSnapshot
from google.cloud.firestore_v1.vector import Vector
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

DEFAULT_BATCH_SIZE = 500
DEFAULT_TOP_K = 10

_logger = logging.getLogger(__name__)


# def _to_firestore_filter(standard_filters: MetadataFilters) -> Any:
#     """Convert from standard dataclass to filter dict."""
#     filters = {}
#     for f in standard_filters.filters():

#     return filters


class FirestoreVectorStore(BasePydanticVectorStore):
    """Firestore Vector Store."""

    stores_text: bool = True
    flat_metadata: bool = True

    collection_name: str
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE
    embedding_key: Optional[str] = "embedding"
    text_key: Optional[str] = "text"
    metadata_key: Optional[str] = "metadata"
    distance_strategy: Optional[DistanceMeasure] = DistanceMeasure.COSINE

    _client: Client

    def __init__(
        self,
        client: Optional[Client] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(**kwargs)
        object.__setattr__(self, "_client", client or Client())

    @classmethod
    def class_name(cls) -> str:
        return "FirestoreVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to vector store."""
        ids = []
        entries = []
        for node in nodes:
            node_id = node.node_id
            metadata = node_to_metadata_dict(
                node,
                remove_text=not self.stores_text,
                flat_metadata=self.flat_metadata,
            )
            entry = {
                self.embedding_key: node.get_embedding(),
                self.metadata_key: metadata,
            }
            ids.append(node_id)
            entries.append(entry)
        self._upsert_batch(entries, ids)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id."""
        self._client.collection(self.collection_name).document(ref_doc_id).delete()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        if query.query_embedding is None:
            raise ValueError("Query embedding is required.")

        k = kwargs.get("k") or DEFAULT_TOP_K
        results = self._similarity_search(query.query_embedding, k, **kwargs)

        top_k_ids = []
        top_k_nodes = []

        for result in results:
            result_dict = result.to_dict() or {}
            if self.metadata_key:
                metadata = result_dict.get(self.metadata_key)

            node = metadata_dict_to_node(metadata)

            if result_dict.get(self.text_key) is not None:
                node.set_content(result_dict.get(self.text_key))

            top_k_ids.append(result.id)
            top_k_nodes.append(node)

        final_result = VectorStoreQueryResult(nodes=top_k_nodes, ids=top_k_ids)
        _logger.debug("Result of query: %s", final_result)
        return final_result

    def _upsert_batch(self, entries: List[dict], ids: Optional[List[str]]) -> None:
        """Upsert batch of vectors to Firestore."""
        if ids and len(ids) != len(entries):
            raise ValueError("Length of ids and entries should be the same.")

        db_batch = self._client.batch()

        for batch in more_itertools.chunked(entries, DEFAULT_BATCH_SIZE):
            for i, entry in enumerate(batch):
                # Convert the embedding array to a Firestore Vector
                entry[self.embedding_key] = Vector(entry[self.embedding_key])
                doc = self._client.collection(self.collection_name).document(
                    ids[i] if ids else None
                )
                db_batch.set(doc, entry, merge=True)
            db_batch.commit()

    def _similarity_search(
        self, query: List[float], k: int = DEFAULT_TOP_K, **kwargs: Any
    ) -> List[DocumentSnapshot]:
        _filters = kwargs.get("filters")
        _logger.debug("Querying Firestore with filters: %s", _filters)
        _wfilters = None

        results = self._client.collection(self.collection_name).find_nearest(
            vector_field=self.embedding_key,
            query_vector=Vector(query),
            distance_measure=self.distance_strategy,
            limit=k,
        )

        return results.get()
