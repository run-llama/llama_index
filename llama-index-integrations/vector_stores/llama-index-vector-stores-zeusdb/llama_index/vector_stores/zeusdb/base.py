# llama_index/vector_stores/zeusdb/base.py

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

# Use only TYPE_CHECKING imports to avoid runtime hard deps
if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import (  # type: ignore
        VectorStoreQuery,
        VectorStoreQueryResult,
    )

# Try to import the real base, else provide a minimal stub
try:
    from llama_index.core.vector_stores.types import (  # type: ignore
        BasePydanticVectorStore as _BasePydanticVectorStore,  # type: ignore[assignment]
    )
except Exception:  # pragma: no cover

    class _BasePydanticVectorStore:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


# -----------------------------------------------------------------------
# Enterprise Logging Integration with Safe Fallback
# Works without zeusdb or llama_index.core present
# -----------------------------------------------------------------------
try:
    from zeusdb.logging_config import (  # type: ignore[import]
        get_logger as _get_logger,
    )
    from zeusdb.logging_config import (  # type: ignore[import]
        operation_context as _operation_context,
    )
except Exception:  # fallback for OSS or dev environments
    import logging

    class _StructuredAdapter(logging.LoggerAdapter):
        """
        Move arbitrary kwargs into 'extra' for stdlib logging compatibility.
        """

        def process(
            self,
            msg: str,
            kwargs: MutableMapping[str, Any],
        ) -> tuple[str, MutableMapping[str, Any]]:
            allowed = {"exc_info", "stack_info", "stacklevel", "extra"}
            extra = kwargs.get("extra", {}) or {}
            if not isinstance(extra, dict):
                extra = {"_extra": repr(extra)}
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


# Initialize module logger through central configuration
get_logger: Callable[[str], Any] = cast(Callable[[str], Any], _get_logger)
logger = get_logger("llamaindex_zeusdb")
operation_context = cast(Callable[..., Any], _operation_context)


# -------------------------
# Utilities and type helpers
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
    Convert ZeusDB distance to a similarity score, higher is better.
    cosine: similarity = 1 - distance (assumes normalized embeddings).
    l2 or l1: return negative distance to match higher is better.
    """
    if space == "cosine":
        return 1.0 - float(distance_value)
    return -float(distance_value)


def _extract_embedding(node: Any) -> list[float] | None:
    emb = getattr(node, "embedding", None)
    if emb is None and hasattr(node, "get_embedding"):
        try:
            emb = node.get_embedding()  # type: ignore[attr-defined]
        except Exception:
            emb = None
    return list(emb) if emb is not None else None


def _node_metadata(node: Any) -> dict[str, Any]:
    md = dict(getattr(node, "metadata", {}) or {})
    if getattr(node, "ref_doc_id", None):
        md["ref_doc_id"] = node.ref_doc_id
    if getattr(node, "id_", None):
        md["node_id"] = node.id_
    return md


# Safe enum/filter imports with stubs for offline mode
try:
    from llama_index.core.vector_stores.types import (  # type: ignore
        FilterCondition,  # type: ignore[assignment]
        FilterOperator,  # type: ignore[assignment]
        MetadataFilter,  # type: ignore[assignment]
        MetadataFilters,  # type: ignore[assignment]
    )
except Exception:  # pragma: no cover
    # Minimal safe stubs to keep static analyzers quiet
    class FilterCondition:  # type: ignore[no-redef]
        AND = type("V", (), {"value": "and"})

    class FilterOperator:  # type: ignore[no-redef]
        EQ = "eq"
        NE = "ne"
        GT = "gt"
        LT = "lt"
        GTE = "gte"
        LTE = "lte"
        IN = "in"
        NIN = "nin"
        ANY = "any"
        ALL = "all"
        CONTAINS = "contains"
        TEXT_MATCH = "text_match"
        TEXT_MATCH_INSENSITIVE = "text_match_insensitive"
        IS_EMPTY = "is_empty"

    class MetadataFilter:  # type: ignore[no-redef]
        key: str
        value: Any
        operator: Any

    class MetadataFilters:  # type: ignore[no-redef]
        filters: list[Any]
        condition: Any


def _translate_filter_op(op: Any) -> str:
    """
    Map FilterOperator to ZeusDB operator keys.
    Works with actual enums or fallback string values.
    """
    mapping = {
        getattr(FilterOperator, "EQ", "eq"): "eq",
        getattr(FilterOperator, "NE", "ne"): "ne",
        getattr(FilterOperator, "GT", "gt"): "gt",
        getattr(FilterOperator, "LT", "lt"): "lt",
        getattr(FilterOperator, "GTE", "gte"): "gte",
        getattr(FilterOperator, "LTE", "lte"): "lte",
        getattr(FilterOperator, "IN", "in"): "in",
        getattr(FilterOperator, "NIN", "nin"): "nin",
        getattr(FilterOperator, "ANY", "any"): "any",
        getattr(FilterOperator, "ALL", "all"): "all",
        getattr(FilterOperator, "CONTAINS", "contains"): "contains",
        getattr(FilterOperator, "TEXT_MATCH", "text_match"): "text_match",
        getattr(
            FilterOperator,
            "TEXT_MATCH_INSENSITIVE",
            "text_match_insensitive",
        ): "text_match_insensitive",
        getattr(FilterOperator, "IS_EMPTY", "is_empty"): "is_empty",
    }
    return mapping.get(op, "eq")


def _filters_to_zeusdb(
    filters: MetadataFilters | None,
) -> dict[str, Any] | None:
    """
    Convert LlamaIndex MetadataFilters to ZeusDB flat format.

    ZeusDB expects a flat dict with implicit AND:
    {"key1": value, "key2": {"op": value}}
    """
    if filters is None:
        return None

    def _one(f: MetadataFilter | MetadataFilters) -> dict[str, Any]:
        if isinstance(f, MetadataFilters):
            cond_val = getattr(getattr(f, "condition", None), "value", "and")
            cond = str(cond_val).lower() if cond_val else "and"
            sub = [_one(sf) for sf in getattr(f, "filters", [])]

            if cond == "and":
                merged: dict[str, Any] = {}
                for s in sub:
                    merged.update(s)
                return merged
            logger.warning(
                "OR filters not supported by ZeusDB backend",
                operation="filter_translation",
                condition=cond,
            )
            return sub[0] if sub else {}

        op_key = _translate_filter_op(getattr(f, "operator", "eq"))
        key = getattr(f, "key", "")
        val = getattr(f, "value", None)

        if op_key == "eq":
            return {key: val}
        return {key: {op_key: val}}

    result = _one(filters)
    logger.debug("translated_filters", zeusdb_filter=result)
    return result


# -------------------------
# MMR helpers (opt in only)
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
    Returns indices of selected candidates.
    """
    n = len(cand_vecs)
    if n == 0 or k <= 0:
        return []

    rel = precomputed_qs or [_cosine_sim(query_vec, v) for v in cand_vecs]
    selected: list[int] = []
    remaining = set(range(n))

    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < min(k, n) and remaining:

        def score(i: int) -> float:
            max_div = max(_cosine_sim(cand_vecs[i], cand_vecs[j]) for j in selected)
            return lamb * rel[i] - (1.0 - lamb) * max_div

        nxt = max(remaining, key=score)
        selected.append(nxt)
        remaining.remove(nxt)

    return selected


# -------------------------
# ZeusDB Vector Store
# -------------------------


class ZeusDBVectorStore(_BasePydanticVectorStore):  # type: ignore[misc]
    """
    LlamaIndex VectorStore backed by ZeusDB (umbrella package).

    Behaviors:
      - Expects nodes with precomputed embeddings
      - Stores vectors and metadata only (stores_text=False)
      - Translates MetadataFilters to ZeusDB flat filters
      - Converts distances to similarity scores (higher=better)
      - Supports optional MMR when requested
      - Provides async wrappers via thread offload
    """

    stores_text: bool = False
    is_embedding_query: bool = True

    def __init__(
        self,
        *,
        dim: int | None = None,
        distance: str = "cosine",
        index_type: str = "hnsw",
        index_name: str = "default",
        quantization_config: dict[str, Any] | None = None,
        m: int | None = None,
        ef_construction: int | None = None,
        expected_size: int | None = None,
        zeusdb_index: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(stores_text=False)

        self._space = _infer_space(distance)
        self._index_name = index_name

        if zeusdb_index is not None:
            self._index = zeusdb_index
        else:
            if dim is None:
                raise ValueError("dim is required when not providing zeusdb_index")
            # Defer zeusdb import to runtime
            from zeusdb import VectorDatabase  # type: ignore

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

    def add(self, nodes: Sequence[Any], **kwargs: Any) -> list[str]:
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
                    ids.append("")

            if not vectors:
                logger.debug(
                    "add_vectors no-op",
                    reason="no embeddings",
                )
                return []

            payload: dict[str, Any] = {
                "vectors": vectors,
                "metadatas": metadatas,
            }

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
                inserted=(len(assigned_ids) if assigned_ids else len(vectors)),
                had_all_ids=(provided_count == len(ids)),
            )

            if assigned_ids:
                return assigned_ids
            return [i for i in ids if i]

    # -------------------------
    # Deletion and maintenance
    # -------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        logger.error(
            "delete by ref_doc_id not supported by ZeusDB HNSW backend",
            operation="delete",
            ref_doc_id=ref_doc_id,
        )
        raise NotImplementedError(
            "ZeusDB HNSW backend does not support deletion by "
            "ref_doc_id. Only remove_point() is supported. "
            "Use delete_nodes(node_ids=[...]) instead."
        )

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        if filters:
            logger.error(
                "delete_nodes with filters not supported by ZeusDB HNSW backend",
                operation="delete_nodes",
                has_filters=True,
            )
            raise NotImplementedError(
                "ZeusDB HNSW backend does not support filter based "
                "deletion. Only direct node ID deletion is supported."
            )

        if not node_ids:
            logger.debug("delete_nodes called with no node_ids")
            return

        with operation_context("delete_nodes", node_ids_count=len(node_ids)):
            try:
                success_count = 0
                failed_ids: list[str] = []

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
        with operation_context("clear_index"):
            try:
                if hasattr(self._index, "clear"):
                    self._index.clear()
                    logger.info(
                        "Index cleared",
                        operation="clear_index",
                    )
                else:
                    logger.warning(
                        "clear not available on index",
                        operation="clear_index",
                    )
                    raise NotImplementedError("ZeusDB index does not expose clear")
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

        Kwargs understood:
            mmr (bool): enable MMR reranking, default False
            mmr_lambda (float): trade-off [0,1], default 0.7
            fetch_k (int): candidate pool when MMR on
            ef_search (int): HNSW search breadth
            return_vector (bool): request raw vectors
            auto_fallback (bool): broaden search if results < k
        """
        with operation_context(
            "query",
            has_embedding=bool(query.query_embedding),
        ):
            if not query.query_embedding:
                from llama_index.core.vector_stores.types import (  # type: ignore
                    VectorStoreQueryResult,
                )

                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            want_mmr = False
            mode = getattr(query, "mode", None)
            if mode and str(mode).lower().endswith("mmr"):
                want_mmr = True
            if kwargs.get("mmr", False):
                want_mmr = True
            mmr_threshold = getattr(query, "mmr_threshold", None)
            if mmr_threshold is not None:
                want_mmr = True

            k = int(query.hybrid_top_k or query.similarity_top_k or 1)
            zfilter = _filters_to_zeusdb(getattr(query, "filters", None))
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
                        ef_search=max(
                            500,
                            max(k, fetch_k) * 10,
                        ),
                        return_vector=return_vector,
                    )
                    if isinstance(broader_res, dict) and ("results" in broader_res):
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
                    from llama_index.core.vector_stores.types import (  # type: ignore
                        VectorStoreQueryResult,
                    )

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
                        nodes=None,
                        similarities=sel_sims,
                        ids=sel_ids,
                    )

            ids: list[str] = []
            sims: list[float] = []
            for _id, dist in zip(cand_ids, cand_dists):
                ids.append(_id)
                sims.append(_similarity_from_distance(dist, self._space))

            from llama_index.core.vector_stores.types import (  # type: ignore
                VectorStoreQueryResult,
            )

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
                    self._index.save(  # type: ignore[attr-defined]
                        persist_path
                    )
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
    # Async wrappers
    # -------------------------

    async def async_add(self, nodes: Sequence[Any], **kwargs: Any) -> list[str]:
        return await asyncio.to_thread(self.add, nodes, **kwargs)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        return await asyncio.to_thread(self.query, query, **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        return await asyncio.to_thread(self.delete, ref_doc_id, **kwargs)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        return await asyncio.to_thread(self.delete_nodes, node_ids, filters, **kwargs)

    async def aclear(self) -> None:
        return await asyncio.to_thread(self.clear)

    # -------------------------
    # Factory methods and utils
    # -------------------------

    @classmethod
    def from_nodes(
        cls,
        nodes: list[Any],
        *,
        dim: int | None = None,
        distance: str = "cosine",
        index_type: str = "hnsw",
        **kwargs: Any,
    ) -> ZeusDBVectorStore:
        if not nodes:
            raise ValueError("Cannot create store from empty nodes list")

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
        with operation_context("load_index", path=path):
            from zeusdb import VectorDatabase  # type: ignore

            vdb = VectorDatabase()
            zeusdb_index = vdb.load(path)

            store = cls(zeusdb_index=zeusdb_index, **kwargs)

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
                                "compression_ratio",
                                "N/A",
                            ),
                            subvectors=quant_info.get("subvectors", "N/A"),
                            bits=quant_info.get("bits", "N/A"),
                        )

                    logger.info(
                        "Index will use raw vectors. Search "
                        "accuracy preserved. Memory compression "
                        "unavailable until next release.",
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
        """Return total vectors in index (best-effort)."""
        try:
            if hasattr(self._index, "get_vector_count"):
                return int(
                    self._index.get_vector_count()  # type: ignore
                )
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
        """Save index to disk (best-effort)."""
        try:
            if hasattr(self._index, "save"):
                self._index.save(  # type: ignore[attr-defined]
                    path
                )
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
        """Get human-readable info string about index."""
        try:
            info_str = self._index.info()
            logger.debug(
                "Retrieved index info",
                operation="info",
                info_length=len(info_str),
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

    # ---------------------------------------------------------------
    # Quantization Methods
    # ---------------------------------------------------------------

    def get_training_progress(self) -> float:
        """Get quantization training progress percentage."""
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
        """Check whether quantized search is active."""
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
        """Check whether quantization is available."""
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
        """Get current storage mode."""
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

    def get_quantization_info(
        self,
    ) -> dict[str, Any] | None:
        """Get detailed quantization information."""
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
