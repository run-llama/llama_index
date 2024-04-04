"""Firestore vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, List, Optional, Union

import more_itertools
from google.cloud.firestore import Client, And, FieldFilter, Or
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.document import DocumentSnapshot
from google.cloud.firestore_v1.vector import Vector
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from .utils import client_with_user_agent

DEFAULT_BATCH_SIZE = 500
DEFAULT_TOP_K = 10

_logger = logging.getLogger(__name__)


def _to_firestore_operator(
    operator: FilterOperator,
) -> str:
    """Convert from standard operator to Firestore operator."""
    if operator == FilterOperator.EQ:
        return "=="
    if operator == FilterOperator.NE:
        return "!="
    if operator == FilterOperator.GT:
        return ">"
    if operator == FilterOperator.GTE:
        return ">="
    if operator == FilterOperator.LT:
        return "<"
    if operator == FilterOperator.LTE:
        return "<="
    if operator == FilterOperator.IN:
        return "in"
    if operator == FilterOperator.NIN:
        return "not-in"
    if operator == FilterOperator.CONTAINS:
        return "array-contains"

    raise ValueError(f"Operator {operator} not supported in Firestore.")


def _to_firestore_filter(
    standard_filters: MetadataFilters,
) -> Union[FieldFilter, Or, And, List[FieldFilter]]:
    """Convert from standard dataclass to filter dict."""
    firestore_filters = []

    # Convert standard filters to Firestore filters
    # Firestore filters are FieldFilter or And/Or filters
    for f in standard_filters.filters:
        if isinstance(f, MetadataFilter):
            firestore_filters.append(
                FieldFilter(f.key, _to_firestore_operator(f.operator), f.value)
            )
        elif isinstance(f, ExactMatchFilter):
            firestore_filters.append(FieldFilter(f.key, "==", f.value))

    if len(firestore_filters) == 1:
        return firestore_filters[0]
    if standard_filters.condition == FilterCondition.AND:
        return And(filters=firestore_filters)
    if standard_filters.condition == FilterCondition.OR:
        return Or(filters=firestore_filters)

    return firestore_filters


class FirestoreVectorStore(BasePydanticVectorStore):
    """Firestore Vector Store."""

    stores_text: bool = True
    flat_metadata: bool = True

    collection_name: str
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE
    embedding_key: str = "embedding"
    text_key: str = "text"
    metadata_key: str = "metadata"
    distance_strategy: DistanceMeasure = DistanceMeasure.COSINE

    _client: Client

    def __init__(
        self,
        client: Optional[Client] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(**kwargs)
        object.__setattr__(self, "_client", client_with_user_agent(client))

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
        """Delete nodes using with ref_doc_id."""
        self._client.collection(self.collection_name).document(ref_doc_id).delete()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        if query.query_embedding is None:
            raise ValueError("Query embedding is required.")

        k = kwargs.get("k") or DEFAULT_TOP_K
        filters = _to_firestore_filter(query.filters) if query.filters else None
        results = self._similarity_search(
            query.query_embedding, k, filters=filters, **kwargs
        )

        top_k_ids = []
        top_k_nodes = []

        for result in results:
            # Convert the Firestore document to dict
            result_dict = result.to_dict() or {}
            metadata = result_dict.get(self.metadata_key) or {}

            # Convert metadata to node, and add text if available
            node = metadata_dict_to_node(metadata, text=result_dict.get(self.text_key))

            # Keep track of the top k ids and nodes
            top_k_ids.append(result.id)
            top_k_nodes.append(node)

        return VectorStoreQueryResult(nodes=top_k_nodes, ids=top_k_ids)

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
        print(_filters)
        _logger.info("Querying Firestore with filters: %s", _filters)

        wfilters = None
        collection = self._client.collection(self.collection_name)

        if _filters is not None:
            wfilters = collection.where(filter=_filters)

        results = (wfilters or collection).find_nearest(
            vector_field=self.embedding_key,
            query_vector=Vector(query),
            distance_measure=self.distance_strategy,
            limit=k,
        )

        return results.get()
