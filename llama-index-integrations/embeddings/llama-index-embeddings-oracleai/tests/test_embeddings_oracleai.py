import os
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.oracleai import OracleEmbeddings

if TYPE_CHECKING:
    import oracledb


def test_class():
    names_of_base_classes = [b.__name__ for b in OracleEmbeddings.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


# unit tests
uname = os.environ.get("VECDB_USER")
passwd = os.environ.get("VECDB_PASS")
v_dsn = os.environ.get("VECDB_HOST")


### Test OracleEmbeddings #####
# @pytest.mark.requires("oracledb")
def test_embeddings_test() -> None:
    try:
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        # print("Connection Successful!")

        doc = """Hello World!!!"""

        # get oracle embeddings
        embedder_params = {"provider": "database", "model": "demo_model"}
        embedder = OracleEmbeddings(conn=connection, params=embedder_params)
        embedding = embedder._get_text_embedding(doc)

        # verify
        assert len(embedding) != 0
        # print(f"Embedding: {embedding}")

        connection.close()
    except Exception as e:
        # print("Error: ", e)
        pass


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
