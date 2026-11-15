import math
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.oracledb import OraLlamaVS
from llama_index.vector_stores.oracledb import base


def test_class():
    names_of_base_classes = [b.__name__ for b in OraLlamaVS.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_query_similarity_decreases_with_distance() -> None:
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    # rows ordered by distance ascending
    cursor.fetchall.return_value = [
        ("closest", "doc", "closest text", "{}", "{}", 0.0),
        ("farther", "doc", "farther text", "{}", "{}", 1.0),
    ]

    with (
        patch.object(base, "_get_connection", return_value=conn),
        patch.object(base, "_create_table"),
    ):
        store = OraLlamaVS(conn, table_name="t")
        result = store.query(
            VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=2)
        )

    assert result.ids == ["closest", "farther"]
    assert result.similarities == pytest.approx([1.0, math.exp(-1.0)])
