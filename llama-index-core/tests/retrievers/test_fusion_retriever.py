import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class MockRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        return self._nodes


@pytest.mark.asyncio
async def test_aretrieve_uses_async_query_generation():
    async_called = []

    class AsyncTrackingLLM(MockLLM):
        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            raise AssertionError("sync complete() must not be called from _aretrieve")

        async def acomplete(self, prompt: str, formatted: bool = False, **kwargs):
            async_called.append(True)
            return CompletionResponse(text="q1\nq2\nq3")

    retriever = QueryFusionRetriever(
        retrievers=[
            MockRetriever([NodeWithScore(node=TextNode(text="result"), score=1.0)])
        ],
        llm=AsyncTrackingLLM(),
        num_queries=4,
    )

    await retriever.aretrieve("test query")

    assert async_called


def _make_retriever(texts):
    return MockRetriever(
        [
            NodeWithScore(node=TextNode(text=t), score=0.9 - i * 0.1)
            for i, t in enumerate(texts)
        ]
    )


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([1.0, 0.0], {"A", "B"}),
        ([0.0, 1.0], {"C", "D"}),
    ],
)
def test_rrf_zero_weight_retriever_does_not_influence_results(weights, expected):
    retriever = QueryFusionRetriever(
        retrievers=[_make_retriever(["A", "B"]), _make_retriever(["C", "D"])],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        retriever_weights=weights,
        similarity_top_k=2,
        num_queries=1,
        use_async=False,
    )
    texts = {n.node.get_content() for n in retriever.retrieve("q")}
    assert texts == expected


def test_rrf_higher_weight_retriever_scores_higher():
    retriever = QueryFusionRetriever(
        retrievers=[
            _make_retriever(["high_A", "high_B"]),
            _make_retriever(["low_A", "low_B"]),
        ],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        retriever_weights=[0.9, 0.1],
        similarity_top_k=4,
        num_queries=1,
        use_async=False,
    )
    scores = {n.node.get_content(): n.score for n in retriever.retrieve("q")}
    assert scores["high_A"] > scores["low_A"]
