# llama_index/vector_stores/zeusdb/base.py

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

# ZeusDB runtime (umbrella package only)
from zeusdb import VectorDatabase  # type: ignore

if TYPE_CHECKING:
    pass

# -----------------------------------------------------------------------------
# Enterprise Logging Integration with Safe Fallback: llamaindex-zeusdb package
# -----------------------------------------------------------------------------
try:
    from zeusdb.logging_config import (  # type: ignore[import]  # noqa: I001
        get_logger as _get_logger,
    )
    from zeusdb.logging_config import (  # type: ignore[import]
        operation_context as _operation_context,
    )
except Exception:  # fallback for OSS/dev environments
    import logging

    class _StructuredAdapter(logging.LoggerAdapter):
        """
        Adapter that moves arbitrary kwargs into 'extra'
        for stdlib logging compatibility.
        """

        def process(
            self, msg: str, kwargs: MutableMapping[str, Any]
        ) -> tuple[str, MutableMapping[str, Any]]:
            allowed = {"exc_info", "stack_info", "stacklevel", "extra"}
            extra = kwargs.get("extra", {}) or {}
            if not isinstance(extra, dict):
                extra = {"_extra": repr(extra)}  # defensive
            fields = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in allowed}
            if fields:
                extra.update(fields)
                kwargs["extra"] = extra
            return msg, kwargs

    def _get_logger(name: str) -> logging.LoggerAdapter:
        base = logging.getLogger(name)
        if not base.handlers:
            base.addHandler(logging.NullHandler())
        return _StructuredAdapter(base, {})

    @contextmanager
    def _operation_context(operation_name: str, **context: Any) -> Iterator[None]:
        logger.debug(
            f"{operation_name} started",
            operation=operation_name,
            **context,
        )
        start = perf_counter()
        try:
            yield
            duration_ms = (perf_counter() - start) * 1000
            logger.info(
                f"{operation_name} completed",
                operation=operation_name,
                duration_ms=duration_ms,
                **context,
            )
        except Exception as e:
            duration_ms = (perf_counter() - start) * 1000
            logger.error(
                f"{operation_name} failed",
                operation=operation_name,
                duration_ms=duration_ms,
                error=str(e),
                exc_info=True,
                **context,
            )
            raise


# Initialize module logger (central config owns handlers/format)
get_logger: Callable[[str], Any] = cast(Callable[[str], Any], _get_logger)
logger = get_logger("llamaindex_zeusdb")
operation_context = cast(Callable[..., Any], _operation_context)

# -------------------------
# Utilities & type helpers
# -------------------------

_DISTANCE_TO_SPACE = {
    "cosine": "cosine",
    "l2": "l2",
    "euclidean": "l2",
    "l1": "l1",
    "manhattan": "l1",
}


def _infer_space(distance: str | None) -> str:
    if not distance:
        return "cosine"
    key = distance.lower()
    return _DISTANCE_TO_SPACE.get(key, "cosine")


def _similarity_from_distance(distance_value: float, space: str) -> float:
    """
    Convert ZeusDB distance to a similarity score (higher = better).
    - cosine: similarity = 1 - distance (assuming normalized embeddings).
    - l2/l1: convert to negative distance so higher is better.
    """
    if space == "cosine":
        return 1.0 - float(distance_value)
    return -float(distance_value)


def _extract_embedding(node: BaseNode) -> list[float] | None:
    # LlamaIndex nodes typically have `embedding` populated before add()
    emb = getattr(node, "embedding", None)
    if emb is None and hasattr(node, "get_embedding"):
        try:
            emb = node.get_embedding()  # type: ignore[attr-defined]
        except Exception:
            emb = None
    return list(emb) if emb is not None else None


def _node_metadata(node: BaseNode) -> dict[str, Any]:
    md = dict(node.metadata or {})
    # Propagate identifiers for delete-by-ref_doc_id and traceability.
    if getattr(node, "ref_doc_id", None):
        md["ref_doc_id"] = node.ref_doc_id
    if getattr(node, "id_", None):
        md["node_id"] = node.id_
    return md


def _translate_filter_op(op: FilterOperator) -> str:
    """Map LlamaIndex FilterOperator -> ZeusDB operator keys."""
    mapping = {
        FilterOperator.EQ: "eq",
        FilterOperator.NE: "ne",
        FilterOperator.GT: "gt",
        FilterOperator.LT: "lt",
        FilterOperator.GTE: "gte",
        FilterOperator.LTE: "lte",
        FilterOperator.IN: "in",
        FilterOperator.NIN: "nin",
        FilterOperator.ANY: "any",
        FilterOperator.ALL: "all",
        FilterOperator.CONTAINS: "contains",
        FilterOperator.TEXT_MATCH: "text_match",
        FilterOperator.TEXT_MATCH_INSENSITIVE: "text_match_insensitive",
        FilterOperator.IS_EMPTY: "is_empty",
    }
    return mapping.get(op, "eq")


def _filters_to_zeusdb(filters: MetadataFilters | None) -> dict[str, Any] | None:
    """
    Convert LlamaIndex MetadataFilters to ZeusDB flat format.

    ZeusDB expects flat dict with implicit AND:
        {"key1": value, "key2": {"op": value}}
    """
    if filters is None:
        return None

    def _one(f: MetadataFilter | MetadataFilters) -> dict[str, Any]:
        if isinstance(f, MetadataFilters):
            cond = (f.condition or FilterCondition.AND).value.lower()
            sub = [_one(sf) for sf in f.filters]

            if cond == "and":
                # Merge into flat dict (implicit AND)
                result = {}
                for s in sub:
                    result.update(s)
                return result
            else:
                # OR is NOT supported by Rust implementation
                logger.warning(
                    "OR filters not supported by ZeusDB backend",
                    operation="filter_translation",
                    condition=cond,
                )
                # Fallback: return first filter only
                return sub[0] if sub else {}

        # Single filter
        op_key = _translate_filter_op(f.operator)

        if op_key == "eq":
            # Direct value for equality (matches Rust code)
            return {f.key: f.value}
        else:
            # Operator wrapper for other ops
            return {f.key: {op_key: f.value}}

    result = _one(filters)  # Changed from 'z' to 'result' for consistency

    logger.debug("translated_filters", zeusdb_filter=result)
    return result


# -------------------------
# MMR helpers (opt-in only)
# -------------------------


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return max(1e-12, sum(x * x for x in a) ** 0.5)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def _mmr_select(
    query_vec: list[float],
    cand_vecs: list[list[float]],
    k: int,
    lamb: float,
    precomputed_qs: list[float] | None = None,
) -> list[int]:
    """
    Greedy Maximal Marginal Relevance.
    Returns indices of the selected candidates.
    """
    n = len(cand_vecs)
    if n == 0 or k <= 0:
        return []

    rel = precomputed_qs or [_cosine_sim(query_vec, v) for v in cand_vecs]
    selected: list[int] = []
    remaining = set(range(n))

    # seed: most relevant
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < min(k, n) and remaining:

        def score(i: int) -> float:
            # diversity term = max sim to any already-selected
            max_div = max(_cosine_sim(cand_vecs[i], cand_vecs[j]) for j in selected)
            return lamb * rel[i] - (1.0 - lamb) * max_div

        nxt = max(remaining, key=score)
        selected.append(nxt)
        remaining.remove(nxt)

    return selected


# -------------------------
# ZeusDB Vector Store
# -------------------------


class ZeusDBVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex VectorStore backed by ZeusDB (via the `zeusdb` umbrella package).

    Behaviors:
      - Expects nodes with precomputed embeddings.
      - Stores vectors + metadata; does not store full text (stores_text=False).
      - Translates LlamaIndex MetadataFilters to ZeusDB filter dicts.
      - Converts ZeusDB distances to similarity scores (higher = better).
      - Supports opt-in MMR when the caller requests it.
      - Provides async wrappers via thread offload.

      Persistence Note: Quantized indexes currently load in raw mode
    """

    stores_text: bool = False
    is_embedding_query: bool = True

    def __init__(
        self,
        *,
        dim: int | None = None,  # Optional if using existing index
        distance: str = "cosine",
        index_type: str = "hnsw",
        index_name: str = "default",
        quantization_config: dict[str, Any] | None = None,
        # ZeusDB tuning params (optional)
        m: int | None = None,
        ef_construction: int | None = None,
        expected_size: int | None = None,
        # Pre-existing ZeusDB index (optional)
        zeusdb_index: Any | None = None,
        # Extra kwargs forwarded to VectorDatabase.create()
        **kwargs: Any,
    ) -> None:
        # super().__init__(stores_text=self.stores_text)
        super().__init__(stores_text=False)  # Use the literal value

        self._space = _infer_space(distance)
        self._index_name = index_name

        if zeusdb_index is not None:
            self._index = zeusdb_index
        else:
            if dim is None:
                raise ValueError("dim is required when not providing zeusdb_index")
            vdb = VectorDatabase()
            create_kwargs: dict[str, Any] = {
                "index_type": index_type,
                "dim": dim,
                "space": self._space,
            }
            if quantization_config is not None:
                create_kwargs["quantization_config"] = quantization_config
            if m is not None:
                create_kwargs["m"] = m
            if ef_construction is not None:
                create_kwargs["ef_construction"] = ef_construction
            if expected_size is not None:
                create_kwargs["expected_size"] = expected_size
            create_kwargs.update(kwargs)
            with operation_context("create_index", space=self._space):
                self._index = vdb.create(**create_kwargs)

    # ---- BasePydanticVectorStore API ----

    @property
    def client(self) -> Any:
        return self._index

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        with operation_context(
            "add_vectors",
            requested=len(nodes),
            overwrite=bool(kwargs.get("overwrite", True)),
        ):
            vectors: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            provided_count = 0

            for n in nodes:
                emb = _extract_embedding(n)
                if emb is None:
                    continue
                vectors.append(emb)
                metadatas.append(_node_metadata(n))
                node_id = getattr(n, "id_", None)
                if node_id is not None:
                    ids.append(str(node_id))
                    provided_count += 1
                else:
                    ids.append("")  # placeholder

            if not vectors:
                logger.debug("add_vectors no-op (no embeddings)")
                return []

            payload: dict[str, Any] = {
                "vectors": vectors,
                "metadatas": metadatas,
            }

            # All-or-nothing ID policy
            if 0 < provided_count < len(ids):
                logger.debug(
                    "partial_ids_ignored",
                    provided_count=provided_count,
                    total=len(ids),
                )
            if provided_count == len(ids):
                payload["ids"] = ids

            overwrite = bool(kwargs.get("overwrite", True))
            try:
                result = self._index.add(payload, overwrite=overwrite)
            except Exception as e:
                logger.error(
                    "ZeusDB add operation failed",
                    operation="add_vectors",
                    node_count=len(nodes),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise

            assigned_ids: list[str] = []
            if isinstance(result, dict) and "ids" in result:
                assigned_ids = [str(x) for x in (result.get("ids") or [])]
            elif hasattr(result, "ids"):
                assigned_ids = [str(x) for x in (getattr(result, "ids") or [])]

            logger.debug(
                "add_vectors summary",
                requested=len(nodes),
                inserted=len(assigned_ids) if assigned_ids else len(vectors),
                had_all_ids=(provided_count == len(ids)),
            )

            # Return backend IDs if available; else fallback to provided ones
            if assigned_ids:
                return assigned_ids
            return [i for i in ids if i]

    # -------------------------
    # Deletion & maintenance
    # -------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete all nodes associated with a ref_doc_id.

        ⚠️  LIMITATION: This method is NOT SUPPORTED by ZeusDB's HNSW backend.

        The HNSW index only supports deletion by node ID via remove_point().
        There is no filter-based deletion or scalable way to find all node IDs
        for a given ref_doc_id (list() doesn't work in QuantizedOnly mode).

        This method will raise NotImplementedError to be honest about the limitation.

        Alternative: Use delete_nodes(node_ids=[...]) if you have the node IDs.
        """
        logger.error(
            "delete() by ref_doc_id is not supported by ZeusDB HNSW backend",
            operation="delete",
            ref_doc_id=ref_doc_id,
        )
        raise NotImplementedError(
            "ZeusDB HNSW backend does not support deletion by ref_doc_id. "
            "The backend only supports ID-based deletion via remove_point(). "
            "Use delete_nodes(node_ids=[...]) instead if you have the node IDs."
        )

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs.

        ✅ SUPPORTED: Deletion by explicit node IDs via remove_point().
        ❌ NOT SUPPORTED: Deletion by metadata filters.

        Args:
            node_ids: List of node IDs to delete (supported)
            filters: Metadata filters (NOT supported - will raise error if provided)

        Note: ZeusDB HNSW only supports direct ID-based deletion.
        """
        if filters:
            logger.error(
                "delete_nodes() with filters is not supported by ZeusDB HNSW backend",
                operation="delete_nodes",
                has_filters=True,
            )
            raise NotImplementedError(
                "ZeusDB HNSW backend does not support filter-based deletion. "
                "Only direct node ID deletion is supported."
            )

        if not node_ids:
            logger.debug("delete_nodes called with no node_ids")
            return

        with operation_context("delete_nodes", node_ids_count=len(node_ids)):
            try:
                success_count = 0
                failed_ids = []

                for node_id in node_ids:
                    try:
                        result = self._index.remove_point(node_id)
                        if result:
                            success_count += 1
                        else:
                            failed_ids.append(node_id)
                    except Exception as e:
                        failed_ids.append(node_id)
                        logger.warning(
                            "Failed to remove point",
                            operation="delete_nodes",
                            node_id=node_id,
                            error=str(e),
                        )

                logger.info(
                    "Delete nodes completed",
                    operation="delete_nodes",
                    requested=len(node_ids),
                    deleted=success_count,
                    failed=len(failed_ids),
                )

                if failed_ids and len(failed_ids) < 10:
                    logger.debug(
                        "Failed node IDs",
                        operation="delete_nodes",
                        failed_ids=failed_ids,
                    )

            except Exception as e:
                logger.error(
                    "Delete nodes failed",
                    operation="delete_nodes",
                    node_ids_count=len(node_ids),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise

    def clear(self) -> None:
        """
        Clear all vectors from the index.

        ⚠️  LIMITATION: May not work correctly in QuantizedOnly mode.

        The clear() method may not properly clear quantized-only vectors.
        """
        with operation_context("clear_index"):
            try:
                if hasattr(self._index, "clear"):
                    self._index.clear()
                    logger.info("Index cleared", operation="clear_index")
                else:
                    logger.warning(
                        "clear() not available on index",
                        operation="clear_index",
                    )
                    raise NotImplementedError(
                        "ZeusDB index does not expose clear() method"
                    )
            except Exception as e:
                logger.error(
                    "Clear operation failed",
                    operation="clear_index",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise

    # -------------------------
    # Query (with optional MMR)
    # -------------------------

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Execute a vector search against ZeusDB.

        Kwargs understood by this adapter:
            mmr (bool): Enable Maximal Marginal Relevance re-ranking. Default False.
            mmr_lambda (float): Trade-off [0..1]. 1=relevance, 0=diversity.
                Default 0.7 (or the provided `query.mmr_threshold`).
            fetch_k (int): Candidate pool when MMR is on. Default max(20, 4*k).
            ef_search (int): HNSW runtime search breadth; forwarded to ZeusDB.
            return_vector (bool): Ask backend to return raw vectors. Auto-enabled
                when MMR is requested.
            auto_fallback (bool): If results < k with no filters, retry once with
                a broader search. Default True.
        """
        with operation_context("query", has_embedding=bool(query.query_embedding)):
            if not query.query_embedding:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            # Detect explicit MMR requests
            want_mmr = False
            mode = getattr(query, "mode", None)
            if mode and str(mode).lower().endswith("mmr"):
                want_mmr = True
            if kwargs.get("mmr", False):
                want_mmr = True
            mmr_threshold = getattr(query, "mmr_threshold", None)
            if mmr_threshold is not None:
                want_mmr = True

            # Common query prep
            k = int(query.hybrid_top_k or query.similarity_top_k or 1)
            zfilter = _filters_to_zeusdb(query.filters)
            ef_search = kwargs.get("ef_search")

            fetch_k = k if not want_mmr else int(kwargs.get("fetch_k", max(20, 4 * k)))
            fetch_k = max(fetch_k, k)

            default_lambda = 0.7 if mmr_threshold is None else mmr_threshold
            mmr_lambda = float(kwargs.get("mmr_lambda", default_lambda))
            if mmr_lambda < 0.0:
                mmr_lambda = 0.0
            elif mmr_lambda > 1.0:
                mmr_lambda = 1.0

            return_vector = bool(kwargs.get("return_vector", False) or want_mmr)

            logger.debug(
                "query parameters",
                k=k,
                fetch_k=fetch_k,
                mmr=want_mmr,
                mmr_lambda=mmr_lambda if want_mmr else None,
                has_filters=zfilter is not None,
                ef_search=ef_search,
                return_vector=return_vector,
            )

            search_kwargs: dict[str, Any] = {
                "vector": list(query.query_embedding),
                "top_k": fetch_k,
            }
            if zfilter is not None:
                search_kwargs["filter"] = zfilter
            if ef_search is not None:
                search_kwargs["ef_search"] = ef_search
            if return_vector:
                search_kwargs["return_vector"] = True

            # Execute search with timing and error context
            t0 = perf_counter()
            try:
                res = self._index.search(**search_kwargs)
            except Exception as e:
                logger.error(
                    "ZeusDB search failed",
                    operation="query",
                    error=str(e),
                    error_type=type(e).__name__,
                    params=search_kwargs,
                    exc_info=True,
                )
                raise
            search_ms = (perf_counter() - t0) * 1000

            # Normalize hits
            hits: list[dict[str, Any]] = []
            if isinstance(res, dict) and "results" in res:
                hits = res.get("results") or []
            elif isinstance(res, list):
                hits = res

            cand_ids: list[str] = []
            cand_dists: list[float] = []
            cand_vecs: list[list[float]] = []

            for h in hits:
                _id = str(h.get("id")) if "id" in h else str(h.get("node_id", ""))
                cand_ids.append(_id)

                if "distance" in h:
                    cand_dists.append(float(h["distance"]))
                elif "score" in h:
                    cand_dists.append(1.0 - float(h["score"]))
                else:
                    cand_dists.append(1.0)

                if return_vector:
                    v = h.get("vector")
                    cand_vecs.append(
                        [float(x) for x in v] if isinstance(v, list) else []
                    )

            # Broadened search fallback (default on)
            fallback_used = False
            if len(cand_ids) < k and not zfilter and kwargs.get("auto_fallback", True):
                logger.debug(
                    "broadening_search_retry",
                    initial_results=len(cand_ids),
                    requested_k=k,
                )
                try:
                    broader_res = self._index.search(
                        vector=list(query.query_embedding),
                        top_k=max(k, fetch_k),
                        ef_search=max(500, max(k, fetch_k) * 10),
                        return_vector=return_vector,
                    )
                    if isinstance(broader_res, dict) and "results" in broader_res:
                        broader_hits = broader_res.get("results") or []
                    elif isinstance(broader_res, list):
                        broader_hits = broader_res
                    else:
                        broader_hits = []

                    if len(broader_hits) > len(hits):
                        hits = broader_hits
                        cand_ids, cand_dists, cand_vecs = [], [], []
                        for h in hits:
                            _id = (
                                str(h.get("id"))
                                if "id" in h
                                else str(h.get("node_id", ""))
                            )
                            cand_ids.append(_id)
                            if "distance" in h:
                                cand_dists.append(float(h["distance"]))
                            elif "score" in h:
                                cand_dists.append(1.0 - float(h["score"]))
                            else:
                                cand_dists.append(1.0)
                            if return_vector:
                                v = h.get("vector")
                                cand_vecs.append(
                                    [float(x) for x in v] if isinstance(v, list) else []
                                )
                        fallback_used = True
                        logger.info(
                            "broadened_search_applied",
                            gained=len(cand_ids),
                        )
                except Exception as e:
                    logger.debug(
                        "broadened_search_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

            # Optional MMR rerank (opt-in only)
            mmr_ms = 0.0
            if want_mmr:
                if (
                    cand_vecs
                    and all(cand_vecs)
                    and isinstance(query.query_embedding, list)
                ):
                    t1 = perf_counter()
                    qv = list(query.query_embedding)
                    rel_q = [_cosine_sim(qv, v) for v in cand_vecs]
                    sel_idx = _mmr_select(
                        qv,
                        cand_vecs,
                        k=k,
                        lamb=mmr_lambda,
                        precomputed_qs=rel_q,
                    )
                    mmr_ms = (perf_counter() - t1) * 1000
                    sel_ids = [cand_ids[i] for i in sel_idx]
                    sel_sims = [rel_q[i] for i in sel_idx]
                    logger.info(
                        "mmr_rerank_applied",
                        selected=len(sel_ids),
                        fetch_k=fetch_k,
                        mmr_lambda=mmr_lambda,
                        search_ms=search_ms,
                        rerank_ms=mmr_ms,
                        space=self._space,
                        fallback_used=fallback_used,
                    )
                    return VectorStoreQueryResult(
                        nodes=None, similarities=sel_sims, ids=sel_ids
                    )
                # If vectors missing, fall through to dense ranking

            # Default: dense similarity ranking
            ids: list[str] = []
            sims: list[float] = []
            for _id, dist in zip(cand_ids, cand_dists):
                ids.append(_id)
                sims.append(_similarity_from_distance(dist, self._space))

            logger.info(
                "Query completed",
                operation="query",
                search_ms=search_ms,
                mmr_ms=mmr_ms,
                results_count=len(hits),
                final_count=len(ids),
                k=k,
                mmr=want_mmr,
                space=self._space,
                fallback_used=fallback_used,
                has_filters=zfilter is not None,
            )
            return VectorStoreQueryResult(
                nodes=None,
                similarities=sims,
                ids=ids,
            )

    def persist(self, persist_path: str, fs: Any | None = None) -> None:
        with operation_context("persist_index", path=persist_path):
            try:
                if hasattr(self._index, "save"):
                    self._index.save(persist_path)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(
                    "ZeusDB persist failed",
                    operation="persist_index",
                    path=persist_path,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise

    # -------------------------
    # Async wrappers (thread offload)
    # -------------------------

    # async def aadd(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        """Thread-offloaded async variant of add()."""
        return await asyncio.to_thread(self.add, nodes, **kwargs)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Thread-offloaded async variant of query()."""
        return await asyncio.to_thread(self.query, query, **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Thread-offloaded async variant of delete()."""
        return await asyncio.to_thread(self.delete, ref_doc_id, **kwargs)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """Thread-offloaded async variant of delete_nodes()."""
        return await asyncio.to_thread(self.delete_nodes, node_ids, filters, **kwargs)

    async def aclear(self) -> None:
        """Thread-offloaded async variant of clear()."""
        return await asyncio.to_thread(self.clear)

    # -------------------------
    # Factory methods and convenience utilities
    # -------------------------

    @classmethod
    def from_nodes(
        cls,
        nodes: list[BaseNode],
        *,
        dim: int | None = None,
        distance: str = "cosine",
        index_type: str = "hnsw",
        **kwargs: Any,
    ) -> ZeusDBVectorStore:
        """Create ZeusDBVectorStore from nodes with embeddings."""
        if not nodes:
            raise ValueError("Cannot create store from empty nodes list")

        # Infer dimension from first node if not provided
        if dim is None:
            first_emb = _extract_embedding(nodes[0])
            if first_emb is None:
                raise ValueError("First node has no embedding to infer dimension")
            dim = len(first_emb)

        store = cls(
            dim=dim,
            distance=distance,
            index_type=index_type,
            **kwargs,
        )
        store.add(nodes)
        return store

    @classmethod
    def load_index(
        cls,
        path: str,
        **kwargs: Any,
    ) -> ZeusDBVectorStore:
        """
        Load ZeusDB index from disk.

        Quantized indexes will load in raw mode.
        The quantization model and training state are preserved, but quantized
        search will not be active until the next ZeusDB release.

        The index will function correctly using raw vectors,
        with full search accuracy but without memory compression benefits.
        """
        with operation_context("load_index", path=path):
            vdb = VectorDatabase()
            zeusdb_index = vdb.load(path)

            store = cls(zeusdb_index=zeusdb_index, **kwargs)

            # Detect and warn about quantization state
            try:
                can_use = store.can_use_quantization()
                is_active = store.is_quantized()
                storage_mode = store.get_storage_mode()

                if can_use and not is_active:
                    logger.warning(
                        "Quantized index loaded in raw mode",
                        operation="load_index",
                        storage_mode=storage_mode,
                        can_use_quantization=can_use,
                        is_quantized=is_active,
                    )

                    quant_info = store.get_quantization_info()
                    if quant_info:
                        logger.info(
                            "Quantization config preserved but not active",
                            operation="load_index",
                            compression_ratio=quant_info.get(
                                "compression_ratio", "N/A"
                            ),
                            subvectors=quant_info.get("subvectors", "N/A"),
                            bits=quant_info.get("bits", "N/A"),
                        )

                    logger.info(
                        "Index will use raw vectors. Search accuracy preserved. "
                        "Memory compression unavailable until next release.",
                        operation="load_index",
                    )

            except Exception as e:
                logger.debug(
                    "Could not check quantization status",
                    operation="load_index",
                    error=str(e),
                    error_type=type(e).__name__,
                )

            return store

    def get_vector_count(self) -> int:
        """Return total vectors in the index (best-effort)."""
        try:
            if hasattr(self._index, "get_vector_count"):
                return int(self._index.get_vector_count())  # type: ignore
        except Exception as e:
            logger.error(
                "get_vector_count failed",
                error=str(e),
                error_type=type(e).__name__,
            )
        return 0

    def get_zeusdb_stats(self) -> dict[str, Any]:
        """Return ZeusDB stats (best-effort)."""
        try:
            if hasattr(self._index, "get_stats"):
                stats = self._index.get_stats()  # type: ignore
                return dict(stats) if isinstance(stats, dict) else {}
        except Exception as e:
            logger.error(
                "get_zeusdb_stats failed",
                error=str(e),
                error_type=type(e).__name__,
            )
        return {}

    def save_index(self, path: str) -> bool:
        """Save index to disk (best-effort wrapper)."""
        try:
            if hasattr(self._index, "save"):
                self._index.save(path)  # type: ignore[attr-defined]
                return True
        except Exception as e:
            logger.error(
                "save_index failed",
                path=path,
                error=str(e),
                error_type=type(e).__name__,
            )
        return False

    def info(self) -> str:
        """
        Get a human-readable info string about the index.

        Example:
        >>> print(vector_store.info())
        HNSWIndex(dim=1536, space=cosine, vectors=1200, quantized=True, ...)
        """
        try:
            info_str = self._index.info()
            logger.debug(
                "Retrieved index info", operation="info", info_length=len(info_str)
            )
            return info_str
        except Exception as e:
            logger.error(
                "Failed to get index info",
                operation="info",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return f"ZeusDBVectorStore(error: {e})"

    # -------------------------------------------------------------------------
    # Quantization Methods
    # -------------------------------------------------------------------------

    def get_training_progress(self) -> float:
        """
        Get quantization training progress percentage.

        Returns:
            float: Training progress as percentage (0.0 to 100.0).
                Returns 0.0 if quantization is not configured or on error.

        Example:
            >>> progress = vector_store.get_training_progress()
            >>> print(f"Training: {progress:.1f}% complete")
        """
        try:
            progress = self._index.get_training_progress()
            logger.debug(
                "Retrieved training progress",
                operation="get_training_progress",
                progress_percent=progress,
            )
            return progress
        except Exception as e:
            logger.error(
                "Failed to get training progress",
                operation="get_training_progress",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return 0.0

    def is_quantized(self) -> bool:
        """
        Check whether quantized search is currently active.

        Returns:
            bool: True if index is using quantized vectors for search,
            False otherwise or on error.

        Example:
            >>> if vector_store.is_quantized():
            ...     print("Using quantized search")
        """
        try:
            quantized = self._index.is_quantized()
            logger.debug(
                "Retrieved quantization status",
                operation="is_quantized",
                is_quantized=quantized,
            )
            return quantized
        except Exception as e:
            logger.error(
                "Failed to check quantization status",
                operation="is_quantized",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return False

    def can_use_quantization(self) -> bool:
        """
        Check whether quantization is available (e.g., PQ training completed).

        Returns:
            bool: True if quantization is trained and ready to use,
            False otherwise or on error.

        Example:
            >>> if vector_store.can_use_quantization():
            ...     print("Quantization ready")
        """
        try:
            available = self._index.can_use_quantization()
            logger.debug(
                "Retrieved quantization availability",
                operation="can_use_quantization",
                can_use_quantization=available,
            )
            return available
        except Exception as e:
            logger.error(
                "Failed to check quantization availability",
                operation="can_use_quantization",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return False

    def get_storage_mode(self) -> str:
        """
        Get current storage mode.

        Returns:
            str: Storage mode string. Possible values:
                - 'raw_only': Only raw vectors stored
                - 'quantized_only': Only quantized vectors (memory optimized)
                - 'quantized_with_raw': Both quantized and raw vectors
                - 'quantized_active': Quantization is active
                - 'unknown': On error or unable to determine

        Example:
            >>> mode = vector_store.get_storage_mode()
            >>> print(f"Storage mode: {mode}")
        """
        try:
            mode = self._index.get_storage_mode()
            logger.debug(
                "Retrieved storage mode",
                operation="get_storage_mode",
                storage_mode=mode,
            )
            return mode
        except Exception as e:
            logger.error(
                "Failed to get storage mode",
                operation="get_storage_mode",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return "unknown"

    def get_quantization_info(self) -> dict[str, Any] | None:
        """
        Get detailed quantization information.

        Returns:
            Optional[Dict]: Dictionary containing quantization details:
                - compression_ratio: Memory compression factor (e.g., 16.0 for 16x)
                - memory_mb: Estimated memory usage in megabytes
                - subvectors: Number of subvectors used
                - bits: Bits per quantized code
                - trained: Whether training is complete
                - training_size: Number of vectors used for training
                Returns None if quantization is not configured/trained or on error.

        Example:
            >>> info = vector_store.get_quantization_info()
            >>> if info:
            ...     print(f"Compression: {info['compression_ratio']:.1f}x")
            ...     print(f"Memory: {info['memory_mb']:.2f} MB")
        """
        try:
            info = self._index.get_quantization_info()
            logger.debug(
                "Retrieved quantization info",
                operation="get_quantization_info",
                has_quantization=info is not None,
            )
            return info
        except Exception as e:
            logger.error(
                "Failed to get quantization info",
                operation="get_quantization_info",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return None
