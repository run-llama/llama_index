from unittest.mock import patch

import pytest

from llama_index.core.postprocessor import TimeWeightedPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.mark.parametrize("now", [0.0, 3600.0])
@pytest.mark.parametrize("refresh", [False, True])
def test_time_weighted_explicit_clock(now: float, refresh: bool) -> None:
    node = TextNode(text="document", metadata={"__last_accessed__": now - 3600})
    processor = TimeWeightedPostprocessor(
        now=now, time_decay=0.5, time_access_refresh=refresh
    )

    results = processor.postprocess_nodes([NodeWithScore(node=node, score=0.25)])

    assert results[0].score == pytest.approx(0.75)
    assert node.metadata["__last_accessed__"] == (now if refresh else now - 3600)


def test_time_weighted_default_clock() -> None:
    node = TextNode(text="document", metadata={"__last_accessed__": 3600.0})
    processor = TimeWeightedPostprocessor(time_decay=0.5)

    with patch("llama_index.core.postprocessor.node_recency.datetime") as clock:
        clock.now.return_value.timestamp.return_value = 7200.0
        results = processor.postprocess_nodes([NodeWithScore(node=node, score=0.25)])

    assert results[0].score == pytest.approx(0.75)
    assert node.metadata["__last_accessed__"] == 7200.0
