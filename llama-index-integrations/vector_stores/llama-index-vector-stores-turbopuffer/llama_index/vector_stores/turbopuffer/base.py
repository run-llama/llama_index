"""turbopuffer vector store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from turbopuffer import omit
from turbopuffer.lib.namespace import Namespace
from turbopuffer.types import DistanceMetric, NamespaceQueryResponse
from turbopuffer.types.namespace_multi_query_response import (
    Result as MultiQueryResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from turbopuffer.types.custom import Filter

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100

# Keys managed by the integration that must not be overwritten by user metadata.
# text_key is dynamic so it's added at call sites via `_RESERVED_KEYS | {self.text_key}`.
_RESERVED_KEYS = frozenset({"id", "vector", "$dist"})
_METADATA_PREFIX = "_meta_"

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
        if tpuf_condition == "Not":
            # turbopuffer Not takes a single operand: ("Not", filter).
            return ("Not", filters_list[0])
        return filters_list[0]
    else:
        if tpuf_condition == "Not":
            # turbopuffer Not takes a single operand; wrap multiples in And.
            return ("Not", ("And", filters_list))
        return (tpuf_condition, filters_list)


class TurbopufferVectorStore(BasePydanticVectorStore):
    """
    turbopuffer Vector Store.

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
        text_key: Name of the attribute used to store plain text for
            BM25 full-text search. Defaults to "text".

    Examples:
        >>> from turbopuffer import Turbopuffer
        >>> client = Turbopuffer(api_key="...", region="gcp-us-central1")
        >>> ns = client.namespace("my-namespace")
        >>> store = TurbopufferVectorStore(namespace=ns)

    """

    stores_text: bool = True
    flat_metadata: bool = True

    distance_metric: DistanceMetric = "cosine_distance"
    batch_size: int = DEFAULT_BATCH_SIZE
    text_key: str = "text"

    _namespace: Namespace = PrivateAttr()

    def __init__(
        self,
        namespace: Namespace,
        distance_metric: DistanceMetric = "cosine_distance",
        batch_size: int = DEFAULT_BATCH_SIZE,
        text_key: str = "text",
        **kwargs: object,
    ) -> None:
        super().__init__(
            distance_metric=distance_metric,
            batch_size=batch_size,
            text_key=text_key,
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

    def _build_rows(self, nodes: Sequence[BaseNode]) -> list[dict[str, object]]:
        """Convert BaseNode list to turbopuffer row dicts."""
        reserved = _RESERVED_KEYS | {self.text_key}
        rows: list[dict[str, object]] = []
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )

            row: dict[str, object] = {}
            for key, value in metadata.items():
                if key in reserved:
                    logger.warning(
                        "Metadata key %r conflicts with a reserved "
                        "turbopuffer column. Storing as %r.",
                        key,
                        f"{_METADATA_PREFIX}{key}",
                    )
                    row[f"{_METADATA_PREFIX}{key}"] = value
                else:
                    row[key] = value

            row["id"] = node.node_id
            row["vector"] = node.get_embedding()
            row[self.text_key] = node.get_content(metadata_mode=MetadataMode.NONE) or ""
            rows.append(row)
        return rows

    def _parse_query_result(
        self,
        result: NamespaceQueryResponse | MultiQueryResult,
        *,
        is_bm25: bool = False,
    ) -> VectorStoreQueryResult:
        """Parse a turbopuffer query response into VectorStoreQueryResult."""
        top_k_nodes: list[BaseNode] = []
        top_k_ids: list[str] = []
        top_k_scores: list[float] = []

        for row in result.rows or []:
            row_dict = row.to_dict()
            row_id = str(row_dict.get("id", ""))
            dist = float(row_dict.get("$dist", 0.0))

            if is_bm25:
                # BM25 $dist is a relevance score (higher = better).
                similarity = dist
            elif self.distance_metric == "cosine_distance":
                similarity = 1.0 - dist
            else:
                similarity = 1.0 / (1.0 + dist)

            node = self._row_to_node(row_dict, row_id)

            top_k_nodes.append(node)
            top_k_ids.append(row_id)
            top_k_scores.append(similarity)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_result: VectorStoreQueryResult,
        sparse_result: VectorStoreQueryResult,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> VectorStoreQueryResult:
        """
        Fuse dense and sparse results using Reciprocal Rank Fusion.

        Each document's score is the sum of ``1 / (k + rank)`` across
        result lists. This is position-based and avoids score
        normalization issues across different ranking methods.
        """
        dense_nodes = dense_result.nodes or []
        sparse_nodes = sparse_result.nodes or []

        if not dense_nodes and not sparse_nodes:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        if not sparse_nodes:
            return dense_result
        if not dense_nodes:
            return sparse_result

        all_nodes: dict[str, BaseNode] = {}
        scores: dict[str, float] = {}

        for result_nodes in [dense_nodes, sparse_nodes]:
            for rank, node in enumerate(result_nodes, start=1):
                scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (
                    rrf_k + rank
                )
                if node.node_id not in all_nodes:
                    all_nodes[node.node_id] = node

        fused = sorted(
            [(scores[nid], all_nodes[nid]) for nid in all_nodes],
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return VectorStoreQueryResult(
            nodes=[x[1] for x in fused],
            similarities=[x[0] for x in fused],
            ids=[x[1].node_id for x in fused],
        )

    def _row_to_node(self, row_dict: dict[str, object], row_id: str) -> BaseNode:
        """Convert a turbopuffer row dict to a llama_index BaseNode."""
        reserved = _RESERVED_KEYS | {self.text_key}
        attributes: dict[str, object] = {}
        for k, v in row_dict.items():
            if k in reserved:
                continue
            if k.startswith(_METADATA_PREFIX):
                attributes[k[len(_METADATA_PREFIX) :]] = v
            else:
                attributes[k] = v

        text = str(row_dict.get(self.text_key, "") or "")

        try:
            node = metadata_dict_to_node(attributes)
            node.id_ = row_id
        except Exception:
            logger.debug("Failed to parse node metadata, creating basic TextNode.")
            return TextNode(
                text=text,
                id_=row_id,
                metadata={k: v for k, v in attributes.items() if not k.startswith("_")},
            )

        # Restore the plain text that was stripped by remove_text=True
        # during _build_rows.
        if hasattr(node, "text"):
            node.text = text
        return node

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: object,
    ) -> list[str]:
        """Add nodes to turbopuffer namespace."""
        if not nodes:
            return []

        ids: list[str] = []
        for batch_start in range(0, len(nodes), self.batch_size):
            batch_nodes = nodes[batch_start : batch_start + self.batch_size]
            rows = self._build_rows(batch_nodes)
            ids.extend(str(row["id"]) for row in rows)
            self._namespace.write(
                upsert_rows=rows,
                distance_metric=self.distance_metric,
                schema={
                    self.text_key: {
                        "type": "string",
                        "full_text_search": True,
                    }
                },
            )

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: object) -> None:
        """Delete nodes by ref_doc_id."""
        self._namespace.write(
            delete_by_filter=("doc_id", "Eq", ref_doc_id),
        )

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: object,
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
        **kwargs: object,
    ) -> VectorStoreQueryResult:
        """Query for top k most similar nodes."""
        tpuf_filter = _to_turbopuffer_filter(query.filters) if query.filters else None
        filter_arg = tpuf_filter if tpuf_filter is not None else omit

        mode = query.mode

        if mode == VectorStoreQueryMode.HYBRID:
            if query.query_embedding is None or query.query_str is None:
                raise ValueError(
                    "Hybrid search requires both query_embedding and query_str."
                )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k
            response = self._namespace.multi_query(
                queries=[
                    {
                        "rank_by": ("vector", "ANN", list(query.query_embedding)),
                        "top_k": query.similarity_top_k,
                        "filters": filter_arg,
                        "exclude_attributes": ["vector"],
                    },
                    {
                        "rank_by": (self.text_key, "BM25", query.query_str),
                        "top_k": sparse_top_k,
                        "filters": filter_arg,
                        "exclude_attributes": ["vector"],
                    },
                ],
            )
            dense_result = self._parse_query_result(response.results[0])
            sparse_result = self._parse_query_result(response.results[1], is_bm25=True)
            return self._reciprocal_rank_fusion(
                dense_result,
                sparse_result,
                top_k=query.hybrid_top_k or query.similarity_top_k,
            )

        if mode in (VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.SPARSE):
            if query.query_str is None:
                raise ValueError("Text search requires query_str.")
            top_k = query.sparse_top_k or query.similarity_top_k
            result = self._namespace.query(
                rank_by=(self.text_key, "BM25", query.query_str),
                top_k=top_k,
                filters=filter_arg,
                exclude_attributes=["vector"],
            )
            return self._parse_query_result(result, is_bm25=True)

        # Default: dense vector ANN search.
        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        result = self._namespace.query(
            rank_by=("vector", "ANN", list(query.query_embedding)),
            top_k=query.similarity_top_k,
            filters=filter_arg,
            exclude_attributes=["vector"],
        )
        return self._parse_query_result(result)
