from typing import List, Optional

import pytest

from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class FixedRetriever(BaseRetriever):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=self.text), score=1.0)]


@pytest.mark.parametrize("weights", [[], [1.0], [1.0, 0.0, 100.0]])
def test_fusion_rejects_mismatched_weight_count(weights: List[float]) -> None:
    with pytest.raises(ValueError, match="one weight per retriever"):
        QueryFusionRetriever(
            retrievers=[FixedRetriever("first"), FixedRetriever("second")],
            retriever_weights=weights,
            llm=MockLLM(),
        )


@pytest.mark.parametrize(
    ("weights", "expected"), [(None, [0.5, 0.5]), ([2.0, 1.0], [2 / 3, 1 / 3])]
)
def test_fusion_preserves_valid_weights(
    weights: Optional[List[float]], expected: List[float]
) -> None:
    retriever = QueryFusionRetriever(
        retrievers=[FixedRetriever("first"), FixedRetriever("second")],
        retriever_weights=weights,
        llm=MockLLM(),
        num_queries=1,
        mode=FUSION_MODES.RELATIVE_SCORE,
        use_async=False,
    )

    results = retriever.retrieve("query")

    assert [result.node.get_content() for result in results] == ["first", "second"]
    assert [result.score for result in results] == pytest.approx(expected)
