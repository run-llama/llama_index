"""Turbopuffer vector store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from turbopuffer import omit
from turbopuffer.lib.namespace import Namespace
from turbopuffer.types import DistanceMetric, NamespaceQueryResponse, Row

if TYPE_CHECKING:
    from collections.abc import Sequence

    from turbopuffer.types.custom import Filter

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100

# Keys added by turbopuffer in query results that are not user attributes.
_RESERVED_RESULT_KEYS = frozenset({"$dist", "vector"})

# Mapping from llama_index FilterOperator values to turbopuffer operator strings.
_OPERATOR_MAP: dict[str, str] = {
    FilterOperator.EQ: "Eq",
    FilterOperator.NE: "NotEq",
    FilterOperator.GT: "Gt",
    FilterOperator.LT: "Lt",
    FilterOperator.GTE: "Gte",
    FilterOperator.LTE: "Lte",
    FilterOperator.IN: "In",
    FilterOperator.NIN: "NotIn",
    FilterOperator.CONTAINS: "Contains",
    FilterOperator.ANY: "ContainsAny",
    FilterOperator.TEXT_MATCH: "Glob",
    FilterOperator.TEXT_MATCH_INSENSITIVE: "IGlob",
}

_CONDITION_MAP: dict[str, str] = {
    FilterCondition.AND: "And",
    FilterCondition.OR: "Or",
    FilterCondition.NOT: "Not",
}


def _to_turbopuffer_filter(
    standard_filters: MetadataFilters,
) -> Filter | None:
    """
    Convert MetadataFilters to turbopuffer tuple-based filter format.

    Single condition example: ("field", "Eq", value)
    Composed example: ("And", [("field", "Eq", v1), ("field2", "Gt", 5)])
    """
    filters_list: list[Filter] = []
    condition = standard_filters.condition or FilterCondition.AND
    tpuf_condition = _CONDITION_MAP.get(condition)
    if tpuf_condition is None:
        raise ValueError(
            f"Filter condition '{condition}' is not supported by turbopuffer."
        )

    for f in standard_filters.filters:
        if isinstance(f, MetadataFilters):
            sub_filter = _to_turbopuffer_filter(f)
            if sub_filter is not None:
                filters_list.append(sub_filter)
            continue
        if f.operator == FilterOperator.IS_EMPTY:
            filters_list.append((f.key, "Eq", None))
        elif f.operator == FilterOperator.ALL:
            if isinstance(f.value, list):
                sub: list[Filter] = [(f.key, "Contains", v) for v in f.value]
                if len(sub) == 1:
                    filters_list.append(sub[0])
                else:
                    filters_list.append(("And", sub))
            else:
                filters_list.append((f.key, "Contains", f.value))
        else:
            tpuf_op = _OPERATOR_MAP.get(f.operator)
            if tpuf_op is None:
                raise ValueError(
                    f"Filter operator '{f.operator}' is not supported "
                    f"by turbopuffer. Supported: {list(_OPERATOR_MAP.keys())}"
                )
            filters_list.append((f.key, tpuf_op, f.value))

    if len(filters_list) == 0:
        return None
    elif len(filters_list) == 1:
        return filters_list[0]
    else:
        return (tpuf_condition, filters_list)


class TurbopufferVectorStore(BasePydanticVectorStore):
    """
    Turbopuffer Vector Store.

    In this vector store, embeddings and docs are stored within a
    turbopuffer namespace.

    During query time, the index uses turbopuffer to query for the top
    k most similar nodes.

    Args:
        namespace: A turbopuffer ``Namespace`` handle, created via
            ``Turbopuffer(...).namespace("my-ns")``.
        distance_metric: Distance metric for similarity search.
            One of "cosine_distance" or "euclidean_squared".
            Defaults to "cosine_distance".
        batch_size: Batch size for write operations. Defaults to 100.

    Examples:
        >>> from turbopuffer import Turbopuffer
        >>> client = Turbopuffer(api_key="...")
        >>> ns = client.namespace("my-namespace")
        >>> store = TurbopufferVectorStore(namespace=ns)

    """

    stores_text: bool = True
    flat_metadata: bool = True

    distance_metric: DistanceMetric = "cosine_distance"
    batch_size: int = DEFAULT_BATCH_SIZE

    _namespace: Namespace = PrivateAttr()

    def __init__(
        self,
        namespace: Namespace,
        distance_metric: DistanceMetric = "cosine_distance",
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            distance_metric=distance_metric,
            batch_size=batch_size,
            **kwargs,
        )

        self._namespace = namespace

    @classmethod
    def class_name(cls) -> str:
        return "TurbopufferVectorStore"

    @property
    def client(self) -> Namespace:
        """Return the turbopuffer namespace handle."""
        return self._namespace

    def _build_rows(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        """Convert BaseNode list to turbopuffer row dicts."""
        rows: list[dict[str, Any]] = []
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=False, flat_metadata=self.flat_metadata
            )
            row: dict[str, Any] = {
                "id": node.node_id,
                "vector": node.get_embedding(),
            }
            row.update(metadata)
            rows.append(row)
        return rows

    def _parse_query_result(
        self, result: NamespaceQueryResponse
    ) -> VectorStoreQueryResult:
        """Parse a turbopuffer query response into VectorStoreQueryResult."""
        top_k_nodes: list[BaseNode] = []
        top_k_ids: list[str] = []
        top_k_scores: list[float] = []

        for row in result.rows or []:
            row_id = str(row.id)
            dist: float = getattr(row, "$dist", 0.0)

            # Convert distance to similarity score.
            if self.distance_metric == "cosine_distance":
                similarity = 1.0 - dist
            else:
                similarity = 1.0 / (1.0 + dist)

            # Reconstruct node from stored metadata attributes.
            node = self._row_to_node(row, row_id)

            top_k_nodes.append(node)
            top_k_ids.append(row_id)
            top_k_scores.append(similarity)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    @staticmethod
    def _row_to_node(row: Row, row_id: str) -> BaseNode:
        """Convert a turbopuffer Row to a llama_index BaseNode."""
        row_dict = row.to_dict()
        attributes = {
            k: v
            for k, v in row_dict.items()
            if k not in _RESERVED_RESULT_KEYS and k != "id"
        }

        try:
            node = metadata_dict_to_node(attributes)
            node.id_ = row_id
        except Exception:
            logger.debug("Failed to parse node metadata, creating basic TextNode.")
            node = TextNode(
                text=str(attributes.pop("_node_content", "")),
                id_=row_id,
                metadata={k: v for k, v in attributes.items() if not k.startswith("_")},
            )
        return node

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> list[str]:
        """Add nodes to turbopuffer namespace."""
        if not nodes:
            return []

        ids: list[str] = []
        for batch_start in range(0, len(nodes), self.batch_size):
            batch_nodes = nodes[batch_start : batch_start + self.batch_size]
            rows = self._build_rows(batch_nodes)
            ids.extend(row["id"] for row in rows)
            self._namespace.write(
                upsert_rows=rows,
                distance_metric=self.distance_metric,
            )

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id."""
        self._namespace.write(
            delete_by_filter=("doc_id", "Eq", ref_doc_id),
        )

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by IDs or metadata filters."""
        if node_ids:
            self._namespace.write(deletes=node_ids)
        if filters:
            tpuf_filter = _to_turbopuffer_filter(filters)
            if tpuf_filter is not None:
                self._namespace.write(delete_by_filter=tpuf_filter)

    def clear(self) -> None:
        """Delete all vectors in the namespace."""
        self._namespace.delete_all()

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query for top k most similar nodes."""
        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        query_embedding = list(query.query_embedding)

        tpuf_filter = _to_turbopuffer_filter(query.filters) if query.filters else None

        result = self._namespace.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=query.similarity_top_k,
            filters=tpuf_filter if tpuf_filter is not None else omit,
            exclude_attributes=["vector"],
        )
        return self._parse_query_result(result)
