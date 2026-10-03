import json
import os
import uuid
from contextlib import contextmanager

import pytest

try:
    import oracledb  # type: ignore
except Exception:
    oracledb = None  # allow collection even if client not installed

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)
from llama_index.vector_stores.oracledb import OraLlamaVS, DistanceStrategy
from llama_index.vector_stores.oracledb import base as orallamavs
from llama_index.vector_stores.oracledb.hybrid import (
    OracleVectorizerPreference,
    create_hybrid_index,
)
from llama_index.embeddings.oracleai import OracleEmbeddings


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def _connect_or_skip():
    if oracledb is None:
        pytest.skip("oracledb client not installed")
    # Reuse the same VECDB_* names as the rest of this test package.
    username = _env_or_default("VECDB_USER", "")
    password = _env_or_default("VECDB_PASS", "")
    dsn = _env_or_default(
        "VECDB_HOST",
        "",
    )
    try:
        return oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception as exc:
        pytest.skip(f"Could not connect to Oracle: {exc}")


def _load_embed_params_or_skip():
    # Expect JSON in ORACLE_EMBED_PARAMS, e.g.:
    #   {"provider": "database", "model": "MINI_LM_L6_V2"}
    # or an external embedder spec supported by DBMS_VECTOR_CHAIN
    raw = os.getenv("ORACLE_EMBED_PARAMS")
    if raw:
        return json.loads(raw)
    model = os.getenv("ORACLE_EMBEDDING_MODEL", "ALL_MINILM_L12_V2")
    return {"provider": "database", "model": model}


_TABLES_TO_DROP: set[str] = set()


@pytest.fixture(autouse=True)
def drop_tracked_tables():
    yield
    if not _TABLES_TO_DROP:
        return
    conn = _connect_or_skip()
    try:
        for table_name in list(_TABLES_TO_DROP):
            orallamavs.drop_table_purge(conn, table_name)
    finally:
        _TABLES_TO_DROP.clear()
        conn.close()


def _prep_nodes():
    # Same data pattern as other tests in this repo
    text_json_list = [
        {
            "text": (
                "If the answer to any preceding questions is yes, then the database "
                "stops the search and allocates space from the specified tablespace; "
                "otherwise, space is allocated from the database default shared temporary "
                "tablespace."
            ),
            "id_": "cncpt_15.5.3.2.2_P4",
            "embedding": [1.0, 0.0],
            "relationships": "test-0",
            "metadata": {
                "weight": 1.0,
                "rank": "a",
                "la": "hello2",
                "url": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-5387D7B2-C0CA-4C1E-811B-C7EB9B636442",
            },
        },
        {
            "text": (
                "A tablespace can be online (accessible) or offline (not accessible) "
                "whenever the database is open. A tablespace is usually online so that its "
                "data is available to users. The SYSTEM tablespace and temporary tablespaces "
                "cannot be taken offline."
            ),
            "id_": "cncpt_15.5.5_P1",
            "embedding": [0.0, 1.0],
            "relationships": "test-1",
            "metadata": {
                "weight": 1.9,
                "rank": "c",
                "la": [1, 2, 3],
                "url": "docs.oracle",
            },
        },
        {
            "text": (
                "The database stores LOBs differently from other data types. Creating a LOB "
                "column implicitly creates a LOB segment and a LOB index. The tablespace "
                "containing the LOB segment and LOB index, which are always stored together, "
                "may be different from the tablespace containing the table."
            ),
            "id_": "cncpt_22.3.4.3.1_P2",
            "embedding": [1.0, 1.0],
            "relationships": "test-2",
            "metadata": {
                "weight": 3.0,
                "rank": "d",
                "la": None,
                "url": "https://docs.orajcle.com/en/database/oracle/oracle-database/23/cncpt/concepts-for-database-developers.html#GUID-3C50EAB8-FC39-4BB3-B680-4EACCE49E866",
            },
        },
        {
            "text": (
                "The LOB segment stores data in pieces called chunks. A chunk is a logically "
                "contiguous set of data blocks and is the smallest unit of allocation for a LOB. "
                "A row in the table stores a pointer called a LOB locator, which points to the "
                "LOB index. When the table is queried, the database uses the LOB index to quickly "
                "locate the LOB chunks."
            ),
            "id_": "cncpt_22.3.4.3.1_P3",
            "embedding": [2.0, 1.0],
            "relationships": "test-3",
            "metadata": {
                "weight": 4.0,
                "rank": "e",
                "url": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/concepts-for-database-developers.html#GUID-3C50EAB8-FC39-4BB3-B680-4EACCE49E866",
            },
        },
    ]
    nodes = []
    for j in text_json_list:
        relationships = {
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id=j["relationships"])
        }
        nodes.append(
            TextNode(
                text=j["text"],
                id_=j["id_"],
                embedding=j["embedding"],
                relationships=relationships,
                metadata=j["metadata"],
            )
        )
    return nodes


def _create_hybrid_index(conn, vs: OraLlamaVS, embeddings: OracleEmbeddings) -> str:
    idx_name = f"HYB_{uuid.uuid4().hex[:8]}"
    preference = OracleVectorizerPreference.create_preference(
        conn,
        embeddings,
        f"PREF_{uuid.uuid4().hex[:8]}",
    )
    create_hybrid_index(
        client=conn,
        idx_name=idx_name,
        vector_store=vs,
        vectorizer_preference=preference,
        params={},
    )
    vs.set_hybrid_index(idx_name)
    return idx_name


def _setup_vs_and_index(conn, embed_params) -> OraLlamaVS:
    table_name = f"LLM_IDX_{uuid.uuid4().hex[:8]}"
    _TABLES_TO_DROP.add(table_name)
    try:
        vs = OraLlamaVS.from_documents(
            _prep_nodes(),
            table_name=table_name,
            client=conn,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        embeddings = OracleEmbeddings(conn=conn, params=embed_params)
        _create_hybrid_index(conn, vs, embeddings)
        return vs
    except Exception:
        orallamavs.drop_table_purge(conn, table_name)
        _TABLES_TO_DROP.discard(table_name)
        raise


@contextmanager
def _vector_store(conn, embed_params):
    vs = _setup_vs_and_index(conn, embed_params)
    try:
        yield vs
    finally:
        orallamavs.drop_table_purge(conn, vs.table_name)
        _TABLES_TO_DROP.discard(vs.table_name)


def test_hybrid_basic_search():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()

    with _vector_store(conn, embed_params) as vs:
        # Optionally allow additional search params via env
        extra_params_raw = os.getenv("ORACLE_HYBRID_SEARCH_PARAMS")
        if extra_params_raw:
            try:
                vs.hybrid_search_params = json.loads(extra_params_raw)
            except Exception:
                # ignore malformed override in env
                pass

        # Run a basic HYBRID query
        q = VectorStoreQuery(
            query_str="tablespace LOB segment",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
        )
        res = vs.query(q)

        # Basic assertions
        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        assert len(res.nodes) == len(res.ids)
        # At least one node should be related to the topic
        joined_texts = " ".join([n.get_content() or "" for n in res.nodes]).lower()
        assert "tablespace" in joined_texts or "lob" in joined_texts


def test_hybrid_with_metadata_filters():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()

    with _vector_store(conn, embed_params) as vs:
        # Filter: rank IN ["c"] AND url TEXT_MATCH "docs.oracle"
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="rank", value=["c"], operator=FilterOperator.IN),
                MetadataFilter(
                    key="url", value="docs.oracle", operator=FilterOperator.TEXT_MATCH
                ),
            ],
            condition=FilterCondition.AND,
        )

        q = VectorStoreQuery(
            query_str="tablespace database users",
            similarity_top_k=5,
            filters=filters,
            mode=VectorStoreQueryMode.HYBRID,
        )
        res = vs.query(q)

        # Expect results to respect filters
        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        for n in res.nodes:
            # All results should match the filter criteria
            assert n.metadata.get("rank") == "c"
            assert "docs.oracle" in str(n.metadata.get("url")).lower()


def test_hybrid_with_doc_id_filtering():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()

    with _vector_store(conn, embed_params) as vs:
        # Restrict to a specific doc_id (from relationships 'test-1')
        q = VectorStoreQuery(
            query_str="tablespace",
            similarity_top_k=3,
            doc_ids=["test-1"],
            mode=VectorStoreQueryMode.HYBRID,
        )
        res = vs.query(q)

        # All returned nodes should come from the restricted doc_id path.
        # Since VectorStoreQueryResult doesn't surface doc_id directly,
        # we can at minimum ensure we got results and contents match the target text.
        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        contents = [n.get_content() or "" for n in res.nodes]
        # The target node has "tablespace" and is the one with rank "c"
        assert any("tablespace" in c.lower() for c in contents)


# Additional filtering coverage for all supported operators


def test_hybrid_filter_eq_rank():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(filters=[MetadataFilter(key="rank", value="a")])
    q = VectorStoreQuery(
        query_str="database",
        similarity_top_k=5,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert n.metadata.get("rank") == "a"


def test_hybrid_filter_ne_rank():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="rank", value="a", operator=FilterOperator.NE)]
    )
    q = VectorStoreQuery(
        query_str="database",
        similarity_top_k=5,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert n.metadata.get("rank") != "a"


def test_hybrid_filter_gt_weight():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="weight", value=2.0, operator=FilterOperator.GT)]
    )
    q = VectorStoreQuery(
        query_str="lob",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert float(n.metadata.get("weight", -1)) > 2.0


def test_hybrid_filter_gte_weight():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="weight", value=3.0, operator=FilterOperator.GTE)]
    )
    q = VectorStoreQuery(
        query_str="lob",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert float(n.metadata.get("weight", -1)) >= 3.0


def test_hybrid_filter_lt_weight():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="weight", value=2.0, operator=FilterOperator.LT)]
    )
    q = VectorStoreQuery(
        query_str="tablespace",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert float(n.metadata.get("weight", 999)) < 2.0


def test_hybrid_filter_lte_weight():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="weight", value=1.9, operator=FilterOperator.LTE)]
    )
    q = VectorStoreQuery(
        query_str="tablespace",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert float(n.metadata.get("weight", 999)) <= 1.9


def test_hybrid_filter_in_rank():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", value=["c", "e"], operator=FilterOperator.IN)
        ]
    )
    q = VectorStoreQuery(
        query_str="tablespace",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert n.metadata.get("rank") in {"c", "e"}


"""def test_hybrid_filter_any_array_contains():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    # Node with la [1,2,3] should match ANY of [2, 99]
    filters = MetadataFilters(
        filters=[MetadataFilter(key="la", value=[2, 99], operator=FilterOperator.ANY)]
    )
    q = VectorStoreQuery(
        query_str="tablespace",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        la_val = n.metadata.get("la")
        assert isinstance(la_val, list) and any(v in la_val for v in [2, 99])


def test_hybrid_filter_all_array_contains():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    # Node with la [1,2,3] should contain ALL of [1, 2]
    filters = MetadataFilters(
        filters=[MetadataFilter(key="la", value=[1, 2], operator=FilterOperator.ALL)]
    )
    q = VectorStoreQuery(
        query_str="tablespace",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        la_val = n.metadata.get("la")
        assert isinstance(la_val, list) and set([1, 2]).issubset(set(la_val))
"""


def test_hybrid_filter_text_match_url():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="url", value="docs.oracle", operator=FilterOperator.TEXT_MATCH
            )
        ]
    )
    q = VectorStoreQuery(
        query_str="database",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert "docs.oracle" in str(n.metadata.get("url")).lower()


def test_hybrid_filter_or_condition():
    conn = _connect_or_skip()
    embed_params = _load_embed_params_or_skip()
    vs = _setup_vs_and_index(conn, embed_params)

    # rank == 'a' OR rank == 'c'
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", value="a"),
            MetadataFilter(key="rank", value="c"),
        ],
        condition=FilterCondition.OR,
    )
    q = VectorStoreQuery(
        query_str="database",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vs.query(q)
    assert len(res.ids) >= 1
    for n in res.nodes:
        assert n.metadata.get("rank") in {"a", "c"}
