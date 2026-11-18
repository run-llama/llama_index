from typing import Any, Dict, List, Optional

import pytest
from pytest_mock import MockerFixture

import llama_index.core.vector_stores.simple as simple_module
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode


@pytest.mark.parametrize(
    ("query_threshold", "kwargs", "expected_threshold", "expected_ids"),
    [
        (0.0, {}, 0.0, ["a", "c"]),
        (0.25, {}, 0.25, ["a", "c"]),
        (1.0, {}, 1.0, ["a", "b"]),
        (None, {}, None, ["a", "b"]),
        (None, {"mmr_threshold": 0.0}, 0.0, ["a", "c"]),
        (None, {"mmr_threshold": 1.0}, 1.0, ["a", "b"]),
        (1.0, {"mmr_threshold": 0.0}, 0.0, ["a", "c"]),
        (0.0, {"mmr_threshold": 1.0}, 1.0, ["a", "b"]),
        (0.0, {"mmr_threshold": None}, None, ["a", "b"]),
    ],
)
def test_mmr_threshold_sources(
    query_threshold: Optional[float],
    kwargs: Dict[str, Any],
    expected_threshold: Optional[float],
    expected_ids: List[str],
    mocker: MockerFixture,
) -> None:
    store = SimpleVectorStore()
    store.add(
        [
            TextNode(id_="a", embedding=[1.0, 0.0]),
            TextNode(id_="b", embedding=[1.0, 0.0]),
            TextNode(id_="c", embedding=[0.0, 1.0]),
        ]
    )
    scorer = mocker.spy(simple_module, "get_top_k_mmr_embeddings")

    result = store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
            mmr_threshold=query_threshold,
        ),
        **kwargs,
    )

    assert scorer.call_args.kwargs["mmr_threshold"] == expected_threshold
    assert result.ids == expected_ids
