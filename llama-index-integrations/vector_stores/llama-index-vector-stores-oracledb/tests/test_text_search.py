import os
import uuid
from contextlib import contextmanager
from typing import Any

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
from llama_index.vector_stores.oracledb.text import create_text_index


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def _connect_or_skip() -> Any:
    if oracledb is None:
        pytest.skip("oracledb client not installed")
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


_TABLES_TO_DROP: set[str] = set()


@pytest.fixture(autouse=True)
def drop_tracked_tables() -> None:
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


def _prep_nodes() -> list:
    # Mirror dataset used across integration tests for consistency
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


def _setup_vs_and_text_index(conn: Any, use_fuzzy: bool = False) -> OraLlamaVS:
    table_name = f"LLM_IDX_{uuid.uuid4().hex[:8]}"
    _TABLES_TO_DROP.add(table_name)
    try:
        if use_fuzzy:
            vs = OraLlamaVS(
                _client=conn,
                table_name=table_name,
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
                use_fuzzy_text_search=True,
            )
            vs.add(_prep_nodes())
        else:
            vs = OraLlamaVS.from_documents(
                _prep_nodes(),
                table_name=table_name,
                client=conn,
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
            )
        idx_name = f"TXT_{uuid.uuid4().hex[:8]}"
        create_text_index(client=conn, idx_name=idx_name, vector_store=vs)
        return vs
    except Exception:
        orallamavs.drop_table_purge(conn, table_name)
        _TABLES_TO_DROP.discard(table_name)
        raise


@contextmanager
def _text_vector_store(conn: Any, use_fuzzy: bool = False):
    vs = _setup_vs_and_text_index(conn, use_fuzzy=use_fuzzy)
    try:
        yield vs
    finally:
        orallamavs.drop_table_purge(conn, vs.table_name)
        _TABLES_TO_DROP.discard(vs.table_name)


def test_text_basic_search() -> None:
    conn = _connect_or_skip()
    with _text_vector_store(conn) as vs:
        q = VectorStoreQuery(
            query_str="tablespace LOB segment",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        res = vs.query(q)

        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        # Check topic relevance appears in returned content
        joined_texts = " ".join([n.get_content() or "" for n in res.nodes]).lower()
        assert (
            "tablespace" in joined_texts
            or "lob" in joined_texts
            or "segment" in joined_texts
        )


def test_text_with_metadata_filters() -> None:
    conn = _connect_or_skip()
    with _text_vector_store(conn) as vs:
        # rank IN ["c"] AND url TEXT_MATCH "docs.oracle"
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
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        res = vs.query(q)

        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        for n in res.nodes:
            assert n.metadata.get("rank") == "c"
            assert "docs.oracle" in str(n.metadata.get("url")).lower()


def test_text_with_doc_id_filtering() -> None:
    conn = _connect_or_skip()
    with _text_vector_store(conn) as vs:
        q = VectorStoreQuery(
            query_str="tablespace",
            similarity_top_k=5,
            doc_ids=["test-1"],  # restrict to relationships path used in _prep_nodes
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        res = vs.query(q)

        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        contents = [n.get_content() or "" for n in res.nodes]
        assert any("tablespace" in c.lower() for c in contents)


def test_text_top_k_respects_limit() -> None:
    conn = _connect_or_skip()
    with _text_vector_store(conn) as vs:
        q = VectorStoreQuery(
            query_str="tablespace",
            similarity_top_k=1,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        res = vs.query(q)

        assert isinstance(res.ids, list)
        assert len(res.ids) <= 1


def test_text_fuzzy_token_matching() -> None:
    conn = _connect_or_skip()
    with _text_vector_store(conn, use_fuzzy=True) as vs:
        # Intentionally misspelled token to exercise FUZZY() wrapping
        q = VectorStoreQuery(
            query_str="tablspce",  # close to "tablespace"
            similarity_top_k=5,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        res = vs.query(q)

        assert isinstance(res.ids, list)
        assert len(res.ids) >= 1
        contents = " ".join([n.get_content() or "" for n in res.nodes]).lower()
        # Expect approximate match to still surface docs about "tablespace"
        assert "tablespace" in contents


def test_text_index_idempotent_creation() -> None:
    conn = _connect_or_skip()
    table_name = f"LLM_IDX_{uuid.uuid4().hex[:8]}"
    _TABLES_TO_DROP.add(table_name)
    try:
        vs = OraLlamaVS.from_documents(
            _prep_nodes(),
            table_name=table_name,
            client=conn,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        idx_name = f"TXT_{uuid.uuid4().hex[:8]}"

        # Should not raise on repeated calls
        create_text_index(client=conn, idx_name=idx_name, vector_store=vs)
        create_text_index(client=conn, idx_name=idx_name, vector_store=vs)
    finally:
        orallamavs.drop_table_purge(conn, table_name)
        _TABLES_TO_DROP.discard(table_name)
