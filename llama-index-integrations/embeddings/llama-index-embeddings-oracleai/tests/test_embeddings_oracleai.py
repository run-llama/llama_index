import os
import json
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.oracleai import OracleEmbeddings

try:
    import oracledb  # type: ignore
except Exception:
    oracledb = None  # allow collection even if client not installed


def test_class():
    names_of_base_classes = [b.__name__ for b in OracleEmbeddings.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def _connect_or_skip():
    if oracledb is None:
        pytest.skip("oracledb client not installed")
    # Reuse the same VECDB_* names as the rest of the Oracle integration tests.
    username = _env_or_default("VECDB_USER", "")
    password = _env_or_default("VECDB_PASS", "")
    dsn = _env_or_default("VECDB_HOST", "")
    if not username or not password or not dsn:
        pytest.skip("Oracle test credentials are not set")
    try:
        return oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception as exc:
        pytest.skip(f"Could not connect to Oracle: {exc}")


### Test OracleEmbeddings #####
def test_embeddings_test() -> None:
    connection = _connect_or_skip()
    try:
        doc = """Hello World!!!"""

        # get oracle embeddings
        embedder_params = {"provider": "database", "model": "demo_model"}
        embedder = OracleEmbeddings(conn=connection, params=embedder_params)
        embedding = embedder._get_text_embedding(doc)

        # verify
        assert len(embedding) != 0
        # print(f"Embedding: {embedding}")
    finally:
        connection.close()


# test embedder
# test_embeddings_test()


def _proxy_bind_values(cursor: MagicMock) -> list[str | None]:
    return [
        call.kwargs.get("proxy")
        for call in cursor.execute.call_args_list
        if "utl_http.set_proxy" in call.args[0]
    ]


def test_get_text_embedding_clears_session_proxy_after_success() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (
        json.dumps({"embed_vector": json.dumps([1.0, 2.0])}),
    )
    cursor.execute.side_effect = [None, None, None]

    embedder = OracleEmbeddings(
        conn=conn,
        params={"provider": "database", "model": "demo_model"},
        proxy="http://proxy.example:80",
    )

    with patch("oracledb.defaults"):
        assert embedder._get_text_embedding("hello") == [1.0, 2.0]

    assert _proxy_bind_values(cursor) == ["http://proxy.example:80", None]
    cursor.close.assert_called_once()


def test_get_text_embedding_clears_session_proxy_after_failure() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute.side_effect = [
        None,
        RuntimeError("embedding failed"),
        None,
    ]

    embedder = OracleEmbeddings(
        conn=conn,
        params={"provider": "database", "model": "demo_model"},
        proxy="http://proxy.example:80",
    )

    with patch("oracledb.defaults"):
        with pytest.raises(RuntimeError, match="embedding failed"):
            embedder._get_text_embedding("hello")

    assert _proxy_bind_values(cursor) == ["http://proxy.example:80", None]
    cursor.close.assert_called_once()


def test_get_text_embedding_raises_when_success_cleanup_fails() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (
        json.dumps({"embed_vector": json.dumps([1.0, 2.0])}),
    )
    cursor.execute.side_effect = [None, None, RuntimeError("cleanup failed")]

    embedder = OracleEmbeddings(
        conn=conn,
        params={"provider": "database", "model": "demo_model"},
        proxy="http://proxy.example:80",
    )

    with patch("oracledb.defaults"):
        with pytest.raises(
            RuntimeError,
            match="Failed to clear Oracle session proxy after _get_embedding succeeded",
        ):
            embedder._get_text_embedding("hello")

    assert _proxy_bind_values(cursor) == ["http://proxy.example:80", None]
    cursor.close.assert_called_once()


def test_get_embeddings_raises_when_success_cleanup_fails() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    vector_array_type = MagicMock()
    conn.cursor.return_value = cursor
    conn.gettype.return_value = vector_array_type
    vector_array_type.newobject.return_value = object()
    cursor.__iter__.return_value = iter(
        [(json.dumps({"embed_vector": json.dumps([1.0, 2.0])}),)]
    )
    cursor.execute.side_effect = [None, None, RuntimeError("cleanup failed")]

    embedder = OracleEmbeddings(
        conn=conn,
        params={"provider": "database", "model": "demo_model"},
        proxy="http://proxy.example:80",
    )

    with patch("oracledb.defaults"):
        with pytest.raises(
            RuntimeError,
            match="Failed to clear Oracle session proxy after _get_embeddings succeeded",
        ):
            embedder._get_embeddings(["hello"])

    assert _proxy_bind_values(cursor) == ["http://proxy.example:80", None]
    cursor.close.assert_called_once()
