from typing import Any, Dict, List
import sys
import types

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

if "pyobvector" not in sys.modules:
    pyobvector = types.ModuleType("pyobvector")

    class DummyObVecClient:
        pass

    class DummyVECTOR:
        def __init__(self, dim: int) -> None:
            self.dim = dim

    class DummySparseVector:
        def __call__(self) -> "DummySparseVector":
            return self

    class DummyFtsIndexParam:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class DummyFtsParser:
        NGRAM = "NGRAM"

    class DummyMatchAgainst:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def label(self, *args: Any, **kwargs: Any) -> "DummyMatchAgainst":
            return self

        def desc(self) -> "DummyMatchAgainst":
            return self

    pyobvector.ObVecClient = DummyObVecClient
    pyobvector.VECTOR = DummyVECTOR
    pyobvector.SPARSE_VECTOR = DummySparseVector
    pyobvector.FtsIndexParam = DummyFtsIndexParam
    pyobvector.FtsParser = DummyFtsParser
    pyobvector.MatchAgainst = DummyMatchAgainst

    client_mod = types.ModuleType("pyobvector.client")
    index_param_mod = types.ModuleType("pyobvector.client.index_param")

    class DummyVecIndexType:
        HNSW = "HNSW"
        HNSW_SQ = "HNSW_SQ"
        IVFFLAT = "IVFFLAT"
        IVFSQ = "IVFSQ"
        IVFPQ = "IVFPQ"
        DAAT = "DAAT"

    index_param_mod.VecIndexType = DummyVecIndexType
    client_mod.index_param = index_param_mod

    sys.modules["pyobvector"] = pyobvector
    sys.modules["pyobvector.client"] = client_mod
    sys.modules["pyobvector.client.index_param"] = index_param_mod

from llama_index.vector_stores.oceanbase import base as ob_base


class DummyResult:
    def __init__(self, rows: List[tuple[Any, ...]]):
        self._rows = rows

    def fetchall(self) -> List[tuple[Any, ...]]:
        return self._rows


class DummyClient:
    def __init__(self, rows: List[tuple[Any, ...]] | None = None) -> None:
        self.rows = rows or []
        self.last_ef_search: int | None = None

    def ann_search(self, *args: Any, **kwargs: Any) -> DummyResult:
        return DummyResult(self.rows)

    def set_ob_hnsw_ef_search(self, value: int) -> None:
        self.last_ef_search = value


def make_store() -> ob_base.OceanBaseVectorStore:
    store = ob_base.OceanBaseVectorStore.__new__(ob_base.OceanBaseVectorStore)
    store._metadata_field = "metadata"
    store._doc_id_field = "doc_id"
    store._primary_field = "id"
    store._text_field = "document"
    store._vector_field = "embedding"
    store._sparse_vector_field = "sparse_embedding"
    store._fulltext_field = "fulltext_content"
    store._include_sparse = True
    store._include_fulltext = True
    store._normalize = False
    store._vidx_metric_type = "l2"
    store._index_type = "HNSW"
    store._hnsw_ef_search = -1
    store._table_name = "test_table"
    store._client = DummyClient()
    return store


def test_escape_json_path_segment():
    assert ob_base._escape_json_path_segment("foo") == "foo"
    assert ob_base._escape_json_path_segment("foo-bar") == '"foo-bar"'
    with pytest.raises(ValueError):
        ob_base._escape_json_path_segment("")


def test_enhance_filter_key_quotes_segments():
    store = make_store()
    key = store._enhance_filter_key("foo.bar-baz")
    assert key.startswith("metadata->'$.")
    assert '"bar-baz"' in key


def test_to_oceanbase_filter_builds_params():
    store = make_store()
    params: Dict[str, Any] = {}
    expanding: set[str] = set()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="theme", value="FOO", operator="=="),
            MetadataFilter(key="location", value=[1, 2], operator="in"),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="location", value=None, operator="is_empty"),
                ],
                condition="and",
            ),
            MetadataFilter(key="name", value="bar", operator="text_match"),
        ],
        condition="or",
    )
    clause = store._to_oceanbase_filter(
        filters, params=params, expanding_params=expanding
    )
    assert "metadata" in clause
    assert any(value == "FOO" for value in params.values())
    assert any(value == [1, 2] for value in params.values())
    assert any(value == "bar%" for value in params.values())
    assert any(name.startswith("in_") for name in expanding)


def test_to_oceanbase_filter_not_condition_and_empty_in():
    store = make_store()
    params: Dict[str, Any] = {}
    expanding: set[str] = set()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="score", value=1, operator=">"),
        ],
        condition="not",
    )
    clause = store._to_oceanbase_filter(
        filters, params=params, expanding_params=expanding
    )
    assert clause.startswith("NOT")

    params = {}
    expanding = set()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=[], operator="nin"),
        ]
    )
    clause = store._to_oceanbase_filter(
        filters, params=params, expanding_params=expanding
    )
    assert clause == "1=1"


def test_to_oceanbase_filter_invalid_in_value():
    store = make_store()
    params: Dict[str, Any] = {}
    expanding: set[str] = set()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value="not-a-list", operator="in"),
        ]
    )
    with pytest.raises(ValueError):
        store._to_oceanbase_filter(filters, params=params, expanding_params=expanding)


def test_build_where_clause_with_doc_and_node_ids():
    store = make_store()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="score", value=1, operator=">="),
        ]
    )
    clause = store._build_where_clause(
        filters=filters, doc_ids=["doc-1", "doc-2"], node_ids=["node-1"]
    )
    assert clause is not None
    params = clause.compile().params
    assert any(value == [1] or value == 1 for value in params.values())
    assert ["doc-1", "doc-2"] in params.values()
    assert ["node-1"] in params.values()

    empty_doc_clause = store._build_where_clause(doc_ids=[])
    assert empty_doc_clause is not None
    assert empty_doc_clause.text == "1=0"


def test_normalize_vector():
    assert ob_base._normalize([0.0, 0.0]) == [0.0, 0.0]
    normalized = ob_base._normalize([3.0, 4.0])
    assert normalized == pytest.approx([0.6, 0.8], rel=1e-6)


def test_parse_distance_to_similarity_cosine():
    store = make_store()
    store._vidx_metric_type = "cosine"
    assert store._parse_distance_to_similarities(0.2) == pytest.approx(0.8, rel=1e-6)


def test_parse_metric_type_to_dist_func():
    store = make_store()
    store._vidx_metric_type = "l2"
    assert store._parse_metric_type_str_to_dist_func()(1, 2).name == "l2_distance"
    store._vidx_metric_type = "inner_product"
    assert (
        store._parse_metric_type_str_to_dist_func()(1, 2).name
        == "negative_inner_product"
    )
    store._vidx_metric_type = "cosine"
    assert store._parse_metric_type_str_to_dist_func()(1, 2).name == "cosine_distance"
    store._vidx_metric_type = "invalid"
    with pytest.raises(ValueError):
        store._parse_metric_type_str_to_dist_func()


def test_query_dense_records_and_ef_search():
    store = make_store()
    metadata = node_to_metadata_dict(TextNode(text="text-1"), remove_text=True)
    store._client = DummyClient(
        rows=[
            ("id-1", "text-1", metadata, 0.0),
        ]
    )
    query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=1)
    records = store._query_dense_records(
        query, search_param={"efSearch": 32}, where_clause=None
    )
    assert store._client.last_ef_search == 32
    assert len(records) == 1
    assert records[0]["id"] == "id-1"
    assert records[0]["score"] == pytest.approx(1.0, rel=1e-6)


def test_query_sparse_records_parses_metadata():
    store = make_store()
    metadata = node_to_metadata_dict(TextNode(text="text-1"), remove_text=True)
    store._client = DummyClient(
        rows=[
            ("id-1", "text-1", metadata, 1.0),
        ]
    )
    records = store._query_sparse_records(
        sparse_query={1: 1.0}, top_k=1, where_clause=None
    )
    assert len(records) == 1
    assert records[0]["id"] == "id-1"
    assert records[0]["score"] == pytest.approx(-1.0, rel=1e-6)


def test_query_sparse_and_fulltext_requires_flags():
    store = make_store()
    store._include_sparse = False
    with pytest.raises(ValueError):
        store._query_sparse_records(sparse_query={1: 1.0}, top_k=1, where_clause=None)

    store._include_fulltext = False
    with pytest.raises(ValueError):
        store._query_fulltext_records(fulltext_query="foo", top_k=1, where_clause=None)


def test_hybrid_fusion_weights_and_ranking():
    store = make_store()
    records_by_modality = {
        "vector": [
            {"id": "doc-1", "node": "v1", "score": 0.9, "modality": "vector"},
            {"id": "doc-2", "node": "v2", "score": 0.8, "modality": "vector"},
        ],
        "sparse": [
            {"id": "doc-2", "node": "s2", "score": 0.7, "modality": "sparse"},
        ],
        "fulltext": [
            {"id": "doc-3", "node": "f3", "score": 0.6, "modality": "fulltext"},
        ],
    }
    result = store._fuse_hybrid_records(records_by_modality, top_k=2, alpha=None)
    assert len(result.ids) == 2
    assert result.ids[0] in {"doc-1", "doc-2", "doc-3"}


def test_normalize_hybrid_weights_alpha():
    store = make_store()
    weights = store._normalize_hybrid_weights(["vector", "fulltext"], alpha=0.8)
    assert weights["vector"] == pytest.approx(0.8, rel=1e-6)
    assert weights["fulltext"] == pytest.approx(0.2, rel=1e-6)


def test_query_mode_validations_and_outputs():
    store = make_store()
    store._build_where_clause = lambda *args, **kwargs: None
    store._query_dense_records = lambda *args, **kwargs: [
        {"id": "doc-1", "node": "n1", "score": 0.9, "modality": "vector"}
    ]
    store._query_sparse_records = lambda *args, **kwargs: [
        {"id": "doc-2", "node": "n2", "score": 0.8, "modality": "sparse"}
    ]
    store._query_fulltext_records = lambda *args, **kwargs: [
        {"id": "doc-3", "node": "n3", "score": 0.7, "modality": "fulltext"}
    ]

    with pytest.raises(ValueError):
        store.query(
            VectorStoreQuery(mode=VectorStoreQueryMode.DEFAULT, similarity_top_k=1)
        )

    result = store.query(
        VectorStoreQuery(
            mode=VectorStoreQueryMode.DEFAULT,
            query_embedding=[0.1],
            similarity_top_k=1,
        )
    )
    assert result.ids == ["doc-1"]

    with pytest.raises(ValueError):
        store.query(
            VectorStoreQuery(mode=VectorStoreQueryMode.SPARSE, similarity_top_k=1)
        )

    result = store.query(
        VectorStoreQuery(mode=VectorStoreQueryMode.SPARSE, similarity_top_k=1),
        sparse_query={1: 1.0},
    )
    assert result.ids == ["doc-2"]

    with pytest.raises(ValueError):
        store.query(
            VectorStoreQuery(
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                query_str=None,
                similarity_top_k=1,
            )
        )

    result = store.query(
        VectorStoreQuery(
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            query_str="foo",
            similarity_top_k=1,
        )
    )
    assert result.ids == ["doc-3"]

    with pytest.raises(ValueError):
        store.query(
            VectorStoreQuery(mode=VectorStoreQueryMode.HYBRID, similarity_top_k=1)
        )

    result = store.query(
        VectorStoreQuery(
            mode=VectorStoreQueryMode.HYBRID,
            query_embedding=[0.1],
            query_str="foo",
            similarity_top_k=2,
            hybrid_top_k=1,
        ),
        sparse_query={1: 1.0},
    )
    assert len(result.ids) == 1


def test_init_validations(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyobvector

    class StubClient:
        pass

    monkeypatch.setattr(pyobvector, "ObVecClient", StubClient)
    monkeypatch.setattr(
        ob_base.OceanBaseVectorStore, "_create_table_with_index", lambda self: None
    )

    with pytest.raises(ValueError):
        ob_base.OceanBaseVectorStore(
            client=StubClient(),
            dim=8,
            include_fulltext=True,
            include_sparse=False,
        )

    with pytest.raises(ValueError):
        ob_base.OceanBaseVectorStore(
            client=StubClient(),
            dim=8,
            vidx_metric_type="bad",
        )

    with pytest.raises(ValueError):
        ob_base.OceanBaseVectorStore(
            client=StubClient(),
            dim=8,
            index_type="bad",
        )
