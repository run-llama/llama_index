"""
Pytest bootstrap for ZeusDBVectorStore tests.

Prefers real deps, falls back to lightweight mocks when missing.
"""

from contextlib import contextmanager
import sys
from types import ModuleType

# ----------------------------------------
# Try real llama_index.core first
# ----------------------------------------
try:
    import llama_index.core.vector_stores.types  # type: ignore  # noqa: F401

    _HAS_CORE = True
except Exception:
    _HAS_CORE = False

# ----------------------------------------
# Try real zeusdb first
# ----------------------------------------
try:
    import zeusdb  # type: ignore  # noqa: F401

    _HAS_ZEUSDB = True
except Exception:
    _HAS_ZEUSDB = False

# ----------------------------------------
# Mock llama_index.core types if missing
# ----------------------------------------
if not _HAS_CORE:

    class MockVectorStoreQuery:
        """Mock VectorStoreQuery."""

        def __init__(
            self,
            query_embedding=None,
            similarity_top_k=1,
            hybrid_top_k=None,
            filters=None,
            mode=None,
            mmr_threshold=None,
            query_str=None,
            **kwargs,
        ):
            self.query_embedding = query_embedding
            self.similarity_top_k = similarity_top_k
            self.hybrid_top_k = hybrid_top_k
            self.filters = filters
            self.mode = mode
            self.mmr_threshold = mmr_threshold
            self.query_str = query_str

    class MockVectorStoreQueryResult:
        """Mock VectorStoreQueryResult."""

        def __init__(self, nodes=None, similarities=None, ids=None):
            self.nodes = nodes or []
            self.similarities = similarities or []
            self.ids = ids or []

        def __iter__(self):
            """Match construction style used by code under test."""
            yield from ()

    class MockFilterOperator:
        """Mock FilterOperator enum."""

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

    class MockFilterCondition:
        """Mock FilterCondition enum."""

        class AND:
            value = "and"

    class MockMetadataFilter:
        """Mock MetadataFilter."""

        def __init__(self, key, value, operator):
            self.key = key
            self.value = value
            self.operator = operator

    class MockMetadataFilters:
        """Mock MetadataFilters."""

        def __init__(self, filters=None, condition=None):
            self.filters = filters or []
            self.condition = condition or MockFilterCondition.AND

        @classmethod
        def from_dicts(cls, items, condition=None) -> "MockMetadataFilters":
            """Build MetadataFilters from list of dict specs."""
            fl = []
            for it in items or []:
                fl.append(
                    MockMetadataFilter(
                        key=it.get("key"),
                        value=it.get("value"),
                        operator=it.get("operator", MockFilterOperator.EQ),
                    )
                )
            return cls(filters=fl, condition=condition or MockFilterCondition.AND)

    class MockBasePydanticVectorStore:
        """Mock BasePydanticVectorStore base class."""

        stores_text: bool = False
        is_embedding_query: bool = True

        def __init__(self, stores_text=False, **kwargs):
            self.stores_text = stores_text

    # Build the module tree
    from typing import Any, cast

    core_types: ModuleType = ModuleType("llama_index.core.vector_stores.types")
    ct = cast(Any, core_types)
    ct.VectorStoreQuery = MockVectorStoreQuery
    ct.VectorStoreQueryResult = MockVectorStoreQueryResult
    ct.FilterOperator = MockFilterOperator
    ct.FilterCondition = MockFilterCondition
    ct.MetadataFilter = MockMetadataFilter
    ct.MetadataFilters = MockMetadataFilters
    ct.BasePydanticVectorStore = MockBasePydanticVectorStore

    sys.modules.setdefault("llama_index", ModuleType("llama_index"))
    sys.modules.setdefault("llama_index.core", ModuleType("llama_index.core"))
    sys.modules["llama_index.core.vector_stores"] = ModuleType(
        "llama_index.core.vector_stores"
    )
    sys.modules["llama_index.core.vector_stores.types"] = core_types

# ----------------------------------------
# Mock zeusdb if missing, with minimal behavior
# ----------------------------------------
if not _HAS_ZEUSDB:

    class _InMemoryIndex:
        """Minimal in-memory vector index for CI testing."""

        def __init__(self, *, dim, space, **kwargs):
            self.dim = dim
            self.space = space
            self._vectors: list[list[float]] = []
            self._metadatas: list[dict] = []
            self._ids: list[str] = []
            self._id_counter = 0

        def add(self, payload, overwrite=True):
            """Add vectors to index."""
            vectors = payload.get("vectors") or []
            metadatas = payload.get("metadatas") or [{}] * len(vectors)
            ids = payload.get("ids")
            out_ids = []
            for i, vec in enumerate(vectors):
                if ids and ids[i]:
                    _id = str(ids[i])
                else:
                    self._id_counter += 1
                    _id = str(self._id_counter)
                self._vectors.append(list(vec))
                self._metadatas.append(dict(metadatas[i] if i < len(metadatas) else {}))
                self._ids.append(_id)
                out_ids.append(_id)
            return {"ids": out_ids}

        def _dist(self, a, b):
            """Calculate distance between vectors."""
            if self.space == "cosine":
                import math

                def _dot(x, y):
                    return sum(xx * yy for xx, yy in zip(x, y))

                def _norm(x):
                    return math.sqrt(max(1e-12, sum(xx * xx for xx in x)))

                return 1.0 - (_dot(a, b) / (_norm(a) * _norm(b)))
            # default l2
            import math

            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        def search(
            self,
            *,
            vector,
            top_k,
            filter=None,
            ef_search=None,
            return_vector=False,
        ):
            """Search for similar vectors."""
            results = []
            for _id, v, md in zip(self._ids, self._vectors, self._metadatas):
                # Simple filter: exact match on key: value
                if filter:
                    ok = True
                    for k, val in filter.items():
                        if isinstance(val, dict):
                            if "eq" in val:
                                if md.get(k) != val["eq"]:
                                    ok = False
                                    break
                        else:
                            if md.get(k) != val:
                                ok = False
                                break
                    if not ok:
                        continue
                d = self._dist(vector, v)
                item = {"id": _id, "distance": d}
                if return_vector:
                    item["vector"] = list(v)
                results.append(item)
            results.sort(key=lambda x: x["distance"])
            return results[: int(top_k)]

        def remove_point(self, node_id):
            """Remove a point by ID."""
            try:
                idx = self._ids.index(str(node_id))
            except ValueError:
                return False
            for arr in (self._ids, self._vectors, self._metadatas):
                arr.pop(idx)
            return True

        def clear(self):
            """Clear all vectors."""
            self._vectors.clear()
            self._metadatas.clear()
            self._ids.clear()

        def save(self, path):
            """Save index (no-op for tests)."""
            return True

        def get_vector_count(self):
            """Get number of vectors."""
            return len(self._ids)

        def get_stats(self):
            """Get index statistics."""
            return {"count": len(self._ids)}

        def info(self):
            """Get index info string."""
            return (
                f"HNSWIndex(dim={self.dim}, "
                f"space={self.space}, vectors={len(self._ids)})"
            )

        def get_training_progress(self):
            """Get quantization training progress."""
            return 0.0

        def is_quantized(self):
            """Check if quantized."""
            return False

        def can_use_quantization(self):
            """Check if quantization available."""
            return False

        def get_storage_mode(self):
            """Get storage mode."""
            return "raw_only"

        def get_quantization_info(self):
            """Get quantization info."""
            return None

        def load(self, path):
            """Load index (not used on instance)."""
            return self

    class VectorDatabase:
        """Mock VectorDatabase."""

        def create(self, **kwargs):
            """Create index."""
            return _InMemoryIndex(**kwargs)

        def load(self, path):
            """Load index from path."""
            return _InMemoryIndex(dim=1, space="cosine")

    # Minimal logging_config mock
    logging_config: ModuleType = ModuleType("zeusdb.logging_config")

    def get_logger(name: str):
        """Get logger."""

        class _Dummy:
            def debug(self, *a, **k):
                pass

            def info(self, *a, **k):
                pass

            def warning(self, *a, **k):
                pass

            def error(self, *a, **k):
                pass

        return _Dummy()

    @contextmanager
    def operation_context(operation_name: str, **context):
        """Operation context manager."""
        yield

    logging_config.get_logger = get_logger  # type: ignore[attr-defined]
    logging_config.operation_context = (  # type: ignore[attr-defined]
        operation_context
    )

    zeusdb_mod: ModuleType = ModuleType("zeusdb")
    zeusdb_mod.VectorDatabase = VectorDatabase  # type: ignore[attr-defined]

    sys.modules["zeusdb"] = zeusdb_mod
    sys.modules["zeusdb.logging_config"] = logging_config
