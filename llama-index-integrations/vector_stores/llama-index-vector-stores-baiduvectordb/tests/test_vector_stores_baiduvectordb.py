import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.baiduvectordb import BaiduVectorDB, TableParams


def test_class():
    names_of_base_classes = [b.__name__ for b in BaiduVectorDB.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def _row(node_id: str, text: str, distance: float) -> dict:
    return {
        "distance": distance,
        "row": {
            "id": node_id,
            "text": text,
            "metadata": json.dumps({}),
            "doc_id": node_id,
        },
    }


def _store_with_rows(metric_type: str, rows: list) -> BaiduVectorDB:
    with (
        patch.object(BaiduVectorDB, "_init_client"),
        patch.object(BaiduVectorDB, "_create_database_if_not_exists"),
        patch.object(BaiduVectorDB, "_create_table"),
    ):
        store = BaiduVectorDB(
            endpoint="http://localhost",
            api_key="key",
            table_params=TableParams(dimension=2, metric_type=metric_type),
        )
    table = MagicMock()
    table.search.return_value = SimpleNamespace(rows=rows)
    store._table = table
    return store


def test_l2_distances_are_returned_as_similarities():
    # Mochow returns ascending L2 distances, closest match first.
    store = _store_with_rows("L2", [_row("a", "near", 0.5), _row("b", "far", 2.0)])

    result = asyncio.run(
        store.aquery(VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=2))
    )

    assert result.ids == ["a", "b"]
    assert result.similarities == pytest.approx([1 / 1.5, 1 / 3.0])
    assert result.similarities[0] > result.similarities[1]


@pytest.mark.parametrize("metric_type", ["COSINE", "cosine", "IP"])
def test_similarity_metrics_pass_through(metric_type: str):
    store = _store_with_rows(
        metric_type, [_row("a", "near", 0.9), _row("b", "far", 0.1)]
    )

    result = asyncio.run(
        store.aquery(VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=2))
    )

    assert result.similarities == [0.9, 0.1]
