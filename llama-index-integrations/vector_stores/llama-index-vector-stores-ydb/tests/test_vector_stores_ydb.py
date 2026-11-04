"""
Tests for YDBVectorStore.

Part 1 (unit): no database needed.
Part 2 (integration): need a local YDB, see README.md. If YDB is unreachable the
integration tests are skipped. Connection can be overridden with the environment
variables YDB_HOST, YDB_PORT, YDB_DATABASE.
"""

import asyncio
import contextlib
import json
import math
import os
import uuid
from typing import Any, Dict, List

import pytest
import ydb_dbapi

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.ydb import YDBVectorStore
from llama_index.vector_stores.ydb import base as ydb_base
from llama_index.vector_stores.ydb.base import YDBSearchStrategy


vector_to_bytes = ydb_base._vector_to_bytes
bytes_to_vector = ydb_base._bytes_to_vector

YDB_HOST = os.environ.get("YDB_HOST", "localhost")
YDB_PORT = int(os.environ.get("YDB_PORT", "2136"))
YDB_DATABASE = os.environ.get("YDB_DATABASE", "/local")

# Unit Tests


def _offline_store(**kwargs: Any) -> YDBVectorStore:
    """The constructor does not connect, so this works without YDB."""
    kwargs.setdefault("table_name", "t_offline")
    kwargs.setdefault("embed_dim", 3)
    return YDBVectorStore(**kwargs)


def test_class():
    names_of_base_classes = [b.__name__ for b in YDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_class_name_and_flags():
    assert YDBVectorStore.class_name() == "YDBVectorStore"
    store = _offline_store()
    assert store.stores_text is True
    assert store.is_embedding_query is True


def test_connection_is_lazy():
    assert _offline_store().client is None


def test_from_params_matches_constructor():
    store = YDBVectorStore.from_params(table_name="t_params", embed_dim=5)
    assert store.table_name == "t_params"
    assert store.embed_dim == 5


def test_vector_roundtrip():
    vec = [0.5, -1.0, 2.25]
    data = vector_to_bytes(vec)
    assert data[-1:] == b"\x01"
    assert bytes_to_vector(data) == vec


def test_vector_bytes_layout_is_little_endian_float32():
    data = vector_to_bytes([1.0])
    assert data == b"\x00\x00\x80\x3f\x01"  # 1.0f little-endian + type byte


def test_vector_empty_roundtrip():
    assert bytes_to_vector(vector_to_bytes([])) == []


@pytest.mark.parametrize("bad", [b"", b"\x00\x00\x80\x3f\x02", b"\x00\x00\x80\x01"])
def test_bytes_to_vector_rejects_garbage(bad):
    with pytest.raises(ValueError):
        bytes_to_vector(bad)


# ---- metadata parsing ------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [{"a": 1}, '{"a": 1}', b'{"a": 1}', bytearray(b'{"a": 1}')],
)
def test_parse_metadata_accepts_dict_str_bytes(value):
    assert ydb_base._parse_metadata(value) == {"a": 1}


# ---- strategies ------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (YDBSearchStrategy.COSINE_SIMILARITY, "DESC"),
        (YDBSearchStrategy.INNER_PRODUCT_SIMILARITY, "DESC"),
        (YDBSearchStrategy.COSINE_DISTANCE, "ASC"),
        (YDBSearchStrategy.EUCLIDEAN_DISTANCE, "ASC"),
        (YDBSearchStrategy.MANHATTAN_DISTANCE, "ASC"),
    ],
)
def test_sort_order(strategy, expected):
    assert ydb_base._strategy_sort_order(strategy) == expected


def test_every_strategy_has_index_param():
    for strategy in YDBSearchStrategy:
        param = ydb_base._strategy_index_param(strategy)
        assert param.startswith(("similarity=", "distance="))


@pytest.mark.parametrize(
    ("strategy", "score", "expected"),
    [
        (YDBSearchStrategy.COSINE_SIMILARITY, 0.8, 0.8),
        (YDBSearchStrategy.INNER_PRODUCT_SIMILARITY, 3.0, 3.0),
        (YDBSearchStrategy.COSINE_DISTANCE, 0.25, 0.75),
        (YDBSearchStrategy.EUCLIDEAN_DISTANCE, 2.0, -2.0),
        (YDBSearchStrategy.MANHATTAN_DISTANCE, 5.0, -5.0),
    ],
)
def test_score_to_similarity(strategy, score, expected):
    assert ydb_base._score_to_similarity(strategy, score) == pytest.approx(expected)


def test_score_to_similarity_none_is_zero():
    assert (
        ydb_base._score_to_similarity(YDBSearchStrategy.COSINE_SIMILARITY, None) == 0.0
    )


# ---- validation helpers ----------------------------------------------------


@pytest.mark.parametrize("name", ["abc", "_abc", "a1_b2", "A"])
def test_validate_identifier_ok(name):
    assert ydb_base._validate_identifier(name, "x") == name


@pytest.mark.parametrize(
    "name",
    ["", "1abc", "a-b", "a b", "a`; DROP TABLE x; --", "a/b", "ё", None, 5, "😀"],
)
def test_validate_identifier_rejects(name):
    with pytest.raises(ValueError):
        ydb_base._validate_identifier(name, "x")


@pytest.mark.parametrize("value", [0, -1, 1.5, "3", True, None])
def test_positive_int_rejects(value):
    with pytest.raises(ValueError):
        ydb_base._positive_int(value, "x")


def test_positive_int_ok():
    assert ydb_base._positive_int(7, "x") == 7


def test_chunks():
    assert [list(c) for c in ydb_base._chunks([1, 2, 3, 4, 5], 2)] == [
        [1, 2],
        [3, 4],
        [5],
    ]
    assert list(ydb_base._chunks([], 3)) == []


# ---- constructor validation ------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"table_name": "bad-name"},
        {"table_name": "x`; DROP"},
        {"index_name": "bad name"},
        {"embed_dim": 0},
        {"embed_dim": True},
        {"batch_size": 0},
        {"index_config_levels": 0},
        {"index_config_clusters": -1},
        {"index_tree_search_top_size": 0},
        {"strategy": "NoSuchMetric"},
    ],
)
def test_constructor_rejects_invalid_args(kwargs):
    with pytest.raises(ValueError):
        _offline_store(**kwargs)


def test_strategy_accepts_string():
    store = _offline_store(strategy="EuclideanDistance")
    assert store.strategy is YDBSearchStrategy.EUCLIDEAN_DISTANCE


# ---- node <-> row conversion ----------------------------------------------


def test_node_to_row_and_back():
    node = TextNode(
        id_="n1",
        text="Привет, мир",
        embedding=[1.0, 2.0, 3.0],
        metadata={"lang": "ru", "n": 7},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
    )
    row = ydb_base._node_to_row(node, 3)

    assert row["node_id"] == "n1"
    assert row["ref_doc_id"] == "doc-1"
    assert row["text"] == "Привет, мир"
    assert bytes_to_vector(row["embedding"]) == [1.0, 2.0, 3.0]
    # non-ASCII text must not be stored escaped in the JSON metadata
    assert "\\u" not in row["metadata"]
    # text itself is stored in its own column, not in metadata
    assert "Привет" not in row["metadata"]

    restored = ydb_base._row_to_node({"text": row["text"], "metadata": row["metadata"]})
    assert isinstance(restored, TextNode)
    assert restored.node_id == "n1"
    assert restored.get_content() == "Привет, мир"
    assert restored.metadata == {"lang": "ru", "n": 7}
    assert restored.ref_doc_id == "doc-1"


def test_node_to_row_without_source_has_none_ref_doc_id():
    row = ydb_base._node_to_row(
        TextNode(id_="n", text="t", embedding=[0.0, 0.0, 1.0]), 3
    )
    assert row["ref_doc_id"] is None
    json.loads(row["metadata"])  # valid JSON


def test_node_to_row_checks_dimension():
    node = TextNode(id_="n", text="t", embedding=[1.0, 2.0])
    with pytest.raises(ValueError, match="embed_dim"):
        ydb_base._node_to_row(node, 3)


def test_rows_to_result_converts_scores():
    node = TextNode(id_="a", text="t", embedding=[1.0])
    meta = ydb_base._node_to_row(node, 1)["metadata"]
    rows = [{"node_id": "a", "text": "t", "metadata": meta, "score": 0.25}]

    res = ydb_base._rows_to_result(rows, YDBSearchStrategy.COSINE_DISTANCE)
    assert res.ids == ["a"]
    assert res.similarities == [pytest.approx(0.75)]
    assert res.nodes[0].node_id == "a"


def test_rows_to_nodes_order_dedup_and_embeddings():
    def row(node_id: str, vec: List[float]) -> Dict[str, Any]:
        r = ydb_base._node_to_row(TextNode(id_=node_id, text=node_id, embedding=vec), 2)
        return {
            "node_id": node_id,
            "text": r["text"],
            "metadata": r["metadata"],
            "embedding": r["embedding"],
        }

    rows = [row("a", [1.0, 0.0]), row("b", [0.0, 1.0])]
    nodes = ydb_base._rows_to_nodes(rows, ["b", "missing", "a", "b"])

    assert [n.node_id for n in nodes] == ["b", "a"]
    assert nodes[0].embedding == [0.0, 1.0]
    assert nodes[1].embedding == [1.0, 0.0]


# ---- statement builders ----------------------------------------------------


def test_search_query_without_index():
    store = _offline_store(table_name="tbl")
    q = store._search_query(5)
    assert "VIEW" not in q
    assert "PRAGMA" not in q
    assert "Knn::CosineSimilarity(`embedding`, $embedding)" in q
    assert "ORDER BY score DESC" in q
    assert q.rstrip().endswith("LIMIT 5;")


def test_search_query_with_index_and_distance():
    store = _offline_store(
        table_name="tbl",
        index_enabled=True,
        index_name="idx",
        index_tree_search_top_size=3,
        strategy=YDBSearchStrategy.EUCLIDEAN_DISTANCE,
    )
    q = store._search_query(2)
    assert "VIEW `idx`" in q
    assert 'KMeansTreeSearchTopSize="3"' in q
    assert "Knn::EuclideanDistance" in q
    assert "ORDER BY score ASC" in q


@pytest.mark.parametrize("top_k", [0, -1, None])
def test_search_query_rejects_bad_top_k(top_k):
    with pytest.raises(ValueError):
        _offline_store()._search_query(top_k)


def test_ddl_queries_contain_configuration():
    store = _offline_store(
        table_name="tbl",
        embed_dim=7,
        index_name="idx",
        index_config_levels=3,
        index_config_clusters=64,
        strategy=YDBSearchStrategy.COSINE_DISTANCE,
    )
    create = store._create_table_query()
    assert "CREATE TABLE IF NOT EXISTS `tbl`" in create
    assert "PRIMARY KEY (`node_id`)" in create

    add_index = store._add_vector_index_query()
    assert "`idx__temp`" in add_index
    assert "vector_dimension=7" in add_index
    assert "levels=3" in add_index
    assert "clusters=64" in add_index
    assert "distance=cosine" in add_index

    assert store._drop_temp_index_query() == "ALTER TABLE `tbl` DROP INDEX `idx__temp`;"


def test_ids_params_are_parameters_not_interpolated():
    store = _offline_store()
    nasty = "a'; DROP TABLE x; --"
    id_list, params = store._ids_params([nasty, 5])
    assert id_list == [nasty, "5"]
    assert set(params) == {"$ids"}
    assert params["$ids"][0] == id_list
    assert nasty not in store._get_by_ids_query()
    assert nasty not in store._delete_by_ids_query()


def test_prepare_add_batches():
    store = _offline_store(batch_size=2)
    nodes = [
        TextNode(id_=f"n{i}", text="t", embedding=[0.0, 0.0, 1.0]) for i in range(5)
    ]
    batches = store._prepare_add(nodes)
    assert [len(b) for b in batches] == [2, 2, 1]


# ---- _prepare_search validation -------------------------------------------


def test_prepare_search_ok_and_empty_ids_are_not_a_restriction():
    store = _offline_store()
    # LlamaIndex passes node_ids=[] / doc_ids=[] meaning "no restriction"
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3, node_ids=[], doc_ids=[]
    )
    statement, params = store._prepare_search(q)
    assert "LIMIT 3" in statement
    assert set(params) == {"$embedding"}


@pytest.mark.parametrize(
    "mode",
    [
        VectorStoreQueryMode.HYBRID,
        VectorStoreQueryMode.TEXT_SEARCH,
        VectorStoreQueryMode.SPARSE,
        VectorStoreQueryMode.MMR,
    ],
)
def test_prepare_search_rejects_unsupported_modes(mode):
    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], mode=mode)
    with pytest.raises(NotImplementedError):
        _offline_store()._prepare_search(q)


def test_prepare_search_requires_embedding():
    with pytest.raises(ValueError, match="query_embedding"):
        _offline_store()._prepare_search(VectorStoreQuery(query_str="hi"))


def test_prepare_search_checks_dimension():
    q = VectorStoreQuery(query_embedding=[1.0, 0.0])
    with pytest.raises(ValueError, match="embed_dim"):
        _offline_store()._prepare_search(q)


# Not implemented yet: these tests document the current behavior. When filters /
# node_ids / doc_ids are implemented, replace them with real tests.
def test_prepare_search_filters_not_supported_yet():
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        filters=MetadataFilters(filters=[MetadataFilter(key="a", value=1)]),
    )
    with pytest.raises(NotImplementedError):
        _offline_store()._prepare_search(q)


@pytest.mark.parametrize("field", ["node_ids", "doc_ids"])
def test_prepare_search_id_restrictions_not_supported_yet(field):
    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], **{field: ["x"]})
    with pytest.raises(NotImplementedError):
        _offline_store()._prepare_search(q)


def test_get_nodes_and_delete_nodes_reject_filters_before_connecting():
    store = _offline_store()
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value=1)])
    with pytest.raises(NotImplementedError):
        store.get_nodes(node_ids=["a"], filters=filters)
    with pytest.raises(NotImplementedError):
        store.delete_nodes(node_ids=["a"], filters=filters)
    assert store.client is None  # nothing tried to connect


def test_get_nodes_requires_ids_before_connecting():
    store = _offline_store()
    with pytest.raises(ValueError):
        store.get_nodes(node_ids=None)
    assert store.get_nodes(node_ids=[]) == []
    assert store.client is None


def test_empty_add_and_delete_nodes_do_not_connect():
    store = _offline_store()
    assert store.add([]) == []
    store.delete_nodes(node_ids=[])
    assert store.client is None


##########################################
# Integraion tests on provided from env YDB
##########################################


def _connect():
    return ydb_dbapi.connect(
        host=YDB_HOST, port=YDB_PORT, database=YDB_DATABASE, protocol="grpc"
    )


@pytest.fixture(scope="session")
def ydb_available() -> None:
    try:
        conn = _connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"YDB is not reachable at {YDB_HOST}:{YDB_PORT}: {exc}")
    conn.close()


@pytest.fixture
def make_store(ydb_available: None) -> Any:
    """Factory of stores on unique tables; every table is dropped afterwards."""
    tables: List[str] = []
    stores: List[YDBVectorStore] = []

    def factory(**kwargs: Any) -> YDBVectorStore:
        if "table_name" not in kwargs:
            kwargs["table_name"] = f"t_{uuid.uuid4().hex[:12]}"
        kwargs.setdefault("host", YDB_HOST)
        kwargs.setdefault("port", YDB_PORT)
        kwargs.setdefault("database", YDB_DATABASE)
        kwargs.setdefault("embed_dim", 3)
        if kwargs["table_name"] not in tables:
            tables.append(kwargs["table_name"])
        store = YDBVectorStore(**kwargs)
        stores.append(store)
        return store

    yield factory

    for store in stores:
        try:
            store.close()
        except Exception:
            pass
    conn = _connect()
    try:
        for table in tables:
            with conn.cursor() as cur:
                cur.execute_scheme(f"DROP TABLE IF EXISTS `{table}`")
    finally:
        conn.close()


def _axis_nodes() -> List[TextNode]:
    return [
        TextNode(id_="x", text="along x", embedding=[1.0, 0.0, 0.0]),
        TextNode(id_="xy", text="between", embedding=[1.0, 1.0, 0.0]),
        TextNode(id_="y", text="along y", embedding=[0.0, 1.0, 0.0]),
        TextNode(id_="-x", text="opposite", embedding=[-1.0, 0.0, 0.0]),
    ]


def _count(store: YDBVectorStore) -> int:
    rows = store._execute(f"SELECT COUNT(*) AS c FROM `{store.table_name}`")
    return int(rows[0]["c"])


def _ids(store: YDBVectorStore) -> List[str]:
    rows = store._execute(f"SELECT node_id FROM `{store.table_name}` ORDER BY node_id")
    return [r["node_id"] for r in rows]


def _q(vec: List[float], k: int = 10) -> VectorStoreQuery:
    return VectorStoreQuery(query_embedding=vec, similarity_top_k=k)


@contextlib.asynccontextmanager
async def _aclosing(store: YDBVectorStore):
    try:
        yield store
    finally:
        await store.aclose()


# ---- setup -----------------------------------------------------------------


def test_setup_creates_table_and_is_idempotent(make_store):
    first = make_store()
    assert first.client is None
    first.add(_axis_nodes())
    assert first.client is not None

    # a second store over the same table must not fail or wipe the data
    second = make_store(table_name=first.table_name)
    assert _count(second) == 4


def test_perform_setup_false_does_not_create_table(make_store):
    store = make_store(perform_setup=False)
    with pytest.raises(Exception):
        store.query(_q([1.0, 0.0, 0.0]))


# ---- add / query -----------------------------------------------------------


def test_add_returns_ids_and_writes_rows(make_store):
    store = make_store()
    assert store.add(_axis_nodes()) == ["x", "xy", "y", "-x"]
    assert _ids(store) == ["-x", "x", "xy", "y"]


def test_query_empty_table(make_store):
    res = make_store().query(_q([1.0, 0.0, 0.0]))
    assert res.ids == []
    assert res.nodes == []
    assert res.similarities == []


def test_cosine_similarity_ranking(make_store):
    store = make_store()
    store.add(_axis_nodes())
    res = store.query(_q([1.0, 0.0, 0.0], k=4))

    assert res.ids == ["x", "xy", "y", "-x"]
    assert res.similarities == pytest.approx([1.0, math.sqrt(0.5), 0.0, -1.0], abs=1e-4)
    assert [n.node_id for n in res.nodes] == res.ids


def test_similarity_top_k_limits_results(make_store):
    store = make_store()
    store.add(_axis_nodes())
    assert len(store.query(_q([1.0, 0.0, 0.0], k=2)).ids) == 2


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (YDBSearchStrategy.COSINE_DISTANCE, [1.0, math.sqrt(0.5), 0.0, -1.0]),
        (YDBSearchStrategy.EUCLIDEAN_DISTANCE, [0.0, -1.0, -math.sqrt(2), -2.0]),
    ],
)
def test_distance_strategies_are_converted_to_similarity(
    make_store, strategy, expected
):
    store = make_store(strategy=strategy)
    store.add(_axis_nodes())
    res = store.query(_q([1.0, 0.0, 0.0], k=4))

    assert res.ids == ["x", "xy", "y", "-x"]
    assert res.similarities == pytest.approx(expected, abs=1e-4)


def test_inner_product_strategy(make_store):
    store = make_store(strategy=YDBSearchStrategy.INNER_PRODUCT_SIMILARITY)
    store.add(
        [
            TextNode(id_="big", text="b", embedding=[3.0, 0.0, 0.0]),
            TextNode(id_="small", text="s", embedding=[1.0, 0.0, 0.0]),
            TextNode(id_="neg", text="n", embedding=[-2.0, 0.0, 0.0]),
        ]
    )
    res = store.query(_q([1.0, 0.0, 0.0], k=3))
    assert res.ids == ["big", "small", "neg"]
    assert res.similarities == pytest.approx([3.0, 1.0, -2.0], abs=1e-4)


def test_query_returns_text_and_metadata(make_store):
    store = make_store()
    store.add(
        [
            TextNode(
                id_="n1",
                text="Привет, мир 👋",
                embedding=[1.0, 0.0, 0.0],
                metadata={"lang": "ru", "year": 2024, "tags": ["a", "b"]},
            )
        ]
    )
    node = store.query(_q([1.0, 0.0, 0.0], k=1)).nodes[0]
    assert node.get_content() == "Привет, мир 👋"
    assert node.metadata == {"lang": "ru", "year": 2024, "tags": ["a", "b"]}


def test_add_overwrites_existing_id(make_store):
    store = make_store()
    store.add([TextNode(id_="n", text="old", embedding=[1.0, 0.0, 0.0])])
    store.add([TextNode(id_="n", text="new", embedding=[0.0, 1.0, 0.0])])

    assert _count(store) == 1
    node = store.get_nodes(["n"])[0]
    assert node.get_content() == "new"
    assert node.embedding == [0.0, 1.0, 0.0]


def test_add_in_several_batches(make_store):
    store = make_store(batch_size=2)
    nodes = [
        TextNode(id_=f"n{i}", text=f"t{i}", embedding=[1.0, float(i), 0.0])
        for i in range(5)
    ]
    assert store.add(nodes) == [n.node_id for n in nodes]
    assert _count(store) == 5


def test_add_wrong_dimension_writes_nothing(make_store):
    store = make_store()
    good = TextNode(id_="good", text="g", embedding=[1.0, 0.0, 0.0])
    bad = TextNode(id_="bad", text="b", embedding=[1.0, 0.0])
    with pytest.raises(ValueError):
        store.add([good, bad])
    # conversion happens before the first write, so the table stays empty
    assert _count(store) == 0


def test_query_wrong_dimension(make_store):
    store = make_store()
    with pytest.raises(ValueError):
        store.query(_q([1.0, 0.0]))


def test_query_unsupported_mode(make_store):
    store = make_store()
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], mode=VectorStoreQueryMode.HYBRID
    )
    with pytest.raises(NotImplementedError):
        store.query(q)


# ---- get_nodes -------------------------------------------------------------


def test_get_nodes_order_embeddings_and_missing(make_store):
    store = make_store()
    store.add(_axis_nodes())

    nodes = store.get_nodes(["y", "nope", "x", "y"])
    assert [n.node_id for n in nodes] == ["y", "x"]
    assert nodes[0].embedding == [0.0, 1.0, 0.0]
    assert nodes[1].embedding == [1.0, 0.0, 0.0]
    assert nodes[1].get_content() == "along x"


def test_get_nodes_quote_in_id_is_safe(make_store):
    store = make_store()
    weird = 'it\'s "quoted"; DROP TABLE x'
    store.add([TextNode(id_=weird, text="t", embedding=[1.0, 0.0, 0.0])])
    assert [n.node_id for n in store.get_nodes([weird])] == [weird]
    assert _count(store) == 1


# ---- delete ----------------------------------------------------------------


def test_delete_by_ref_doc_id(make_store):
    store = make_store()

    def node(node_id: str, doc: str) -> TextNode:
        return TextNode(
            id_=node_id,
            text=node_id,
            embedding=[1.0, 0.0, 0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc)},
        )

    store.add([node("a1", "doc-a"), node("a2", "doc-a"), node("b1", "doc-b")])
    store.add([TextNode(id_="orphan", text="o", embedding=[0.0, 1.0, 0.0])])

    store.delete("doc-a")
    assert _ids(store) == ["b1", "orphan"]

    store.delete("unknown-doc")  # no-op, no error
    assert _ids(store) == ["b1", "orphan"]


def test_delete_nodes_by_ids(make_store):
    store = make_store()
    store.add(_axis_nodes())

    store.delete_nodes(node_ids=["x", "y", "nope"])
    assert _ids(store) == ["-x", "xy"]

    store.delete_nodes(node_ids=[])  # no-op
    store.delete_nodes(node_ids=None)  # no-op
    assert _ids(store) == ["-x", "xy"]


def test_clear(make_store):
    store = make_store()
    store.add(_axis_nodes())
    store.clear()
    assert _count(store) == 0
    store.clear()  # clearing an empty table is fine


# ---- vector index ----------------------------------------------------------


def test_rebuild_vector_index_and_query_through_it(make_store):
    # Explicit rebuild creates the index; query then uses the existing index.
    store = make_store(index_enabled=True, index_name="idx_test")
    store.add(_axis_nodes())
    store.rebuild_vector_index()

    indexed = make_store(
        table_name=store.table_name, index_name="idx_test", index_enabled=True
    )
    res = indexed.query(_q([1.0, 0.0, 0.0], k=2))
    assert res.ids[0] == "x"
    assert len(res.ids) == 2

    # Rebuilding an existing index must not fail.
    store.rebuild_vector_index()
    assert indexed.query(_q([1.0, 0.0, 0.0], k=1)).ids == ["x"]


def test_query_creates_missing_vector_index(make_store):
    # With index_enabled=True, query should create the configured index if absent.
    store = make_store(index_enabled=True, index_name="idx_query_auto")
    store.add(_axis_nodes())

    res = store.query(_q([1.0, 0.0, 0.0], k=2))
    assert res.ids == ["x", "xy"]

    # A separate store instance should be able to use the index created by query.
    reopened = make_store(
        table_name=store.table_name,
        index_enabled=True,
        index_name="idx_query_auto",
    )
    assert reopened.query(_q([1.0, 0.0, 0.0], k=1)).ids == ["x"]


# ---- async -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_roundtrip(make_store):
    async with _aclosing(make_store()) as store:
        assert await store.async_add(_axis_nodes()) == ["x", "xy", "y", "-x"]

        res = await store.aquery(_q([1.0, 0.0, 0.0], k=4))
        assert res.ids == ["x", "xy", "y", "-x"]

        got = await store.aget_nodes(["y", "x", "y", "nope"])
        assert [n.node_id for n in got] == ["y", "x"]
        assert got[0].embedding == [0.0, 1.0, 0.0]

        await store.adelete_nodes(node_ids=["x"])
        assert (await store.aget_nodes(["x"])) == []

        await store.adelete_nodes(node_ids=[])  # no-op
        await store.aclear()
        assert (await store.aquery(_q([1.0, 0.0, 0.0]))).ids == []


@pytest.mark.asyncio
async def test_async_delete_by_ref_doc_id(make_store):
    async with _aclosing(make_store()) as store:
        await store.async_add(
            [
                TextNode(
                    id_="a1",
                    text="t",
                    embedding=[1.0, 0.0, 0.0],
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-a")
                    },
                ),
                TextNode(id_="b", text="t", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        await store.adelete("doc-a")
        assert [n.node_id for n in await store.aget_nodes(["a1", "b"])] == ["b"]


@pytest.mark.asyncio
async def test_async_query_creates_missing_vector_index(make_store):
    async with _aclosing(
        make_store(index_enabled=True, index_name="idx_async_query_auto")
    ) as store:
        await store.async_add(_axis_nodes())

        # Async query should create the configured index if it does not exist.
        res = await store.aquery(_q([1.0, 0.0, 0.0], k=2))
        assert res.ids == ["x", "xy"]


@pytest.mark.asyncio
async def test_async_rebuild_vector_index(make_store):
    async with _aclosing(
        make_store(index_enabled=True, index_name="idx_async")
    ) as store:
        await store.async_add(_axis_nodes())
        await store.arebuild_vector_index()

    indexed = make_store(
        table_name=store.table_name, index_name="idx_async", index_enabled=True
    )
    async with _aclosing(indexed):
        res = await indexed.aquery(_q([1.0, 0.0, 0.0], k=1))
        assert res.ids == ["x"]


def test_async_rejects_different_event_loop(make_store):
    """An async store cannot be reused across different event loops."""
    store = make_store()
    store.add(_axis_nodes())

    async def run() -> List[str]:
        return (await store.aquery(_q([1.0, 0.0, 0.0], k=2))).ids

    assert asyncio.run(run()) == ["x", "xy"]

    with pytest.raises(
        RuntimeError,
        match="different event loop",
    ):
        asyncio.run(run())


def test_sync_and_async_see_the_same_data(make_store):
    store = make_store()
    asyncio.run(store.async_add(_axis_nodes()))
    assert _count(store) == 4
    assert store.query(_q([1.0, 0.0, 0.0], k=1)).ids == ["x"]


# ---- LlamaIndex end to end -------------------------------------------------


def test_vector_store_index_end_to_end(make_store):
    dim = 8
    embed_model = MockEmbedding(embed_dim=dim)
    store = make_store(embed_dim=dim)

    docs = [
        Document(id_=f"doc-{i}", text=f"text number {i}", metadata={"i": i})
        for i in range(3)
    ]
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model,
    )
    assert _count(store) == 3

    found = index.as_retriever(similarity_top_k=2).retrieve("anything")
    assert len(found) == 2
    assert all(r.node.get_content().startswith("text number") for r in found)
    assert all("i" in r.node.metadata for r in found)

    index.delete_ref_doc("doc-0")
    assert _count(store) == 2
    remaining = {n.ref_doc_id for n in store.get_nodes(_ids(store))}
    assert remaining == {"doc-1", "doc-2"}

    # the same table can be re-opened as an existing index
    reopened = VectorStoreIndex.from_vector_store(
        make_store(table_name=store.table_name, embed_dim=dim),
        embed_model=embed_model,
    )
    assert len(reopened.as_retriever(similarity_top_k=5).retrieve("anything")) == 2
