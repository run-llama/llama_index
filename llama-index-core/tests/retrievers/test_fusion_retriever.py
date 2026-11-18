import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode


class MockRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text="result"), score=1.0)]


class MockUnscoredRetriever(BaseRetriever):
    """
    Stands in for retrievers that do not assign scores.

    ``SummaryIndexRetriever`` and ``KeywordTableRetriever`` both build
    ``NodeWithScore`` without a score.
    """

    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text="unscored", id_="unscored"))]


class MockMixedRetriever(BaseRetriever):
    """
    Returns a scored node plus an ``IndexNode`` resolved by recursive retrieval.

    ``BaseRetriever._handle_recursive_retrieval`` delegates the ``IndexNode`` to
    the object it maps to and splices the result back in as-is, so the final list
    mixes scored and unscored nodes.
    """

    def _retrieve(self, query_bundle: QueryBundle):
        return [
            NodeWithScore(node=TextNode(text="scored", id_="scored"), score=0.9),
            NodeWithScore(
                node=IndexNode(text="ref", id_="ref", index_id="sub_index"),
                score=0.4,
            ),
        ]


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
        retrievers=[MockRetriever()],
        llm=AsyncTrackingLLM(),
        num_queries=4,
    )

    await retriever.aretrieve("test query")

    assert async_called


@pytest.mark.parametrize("mode", ["relative_score", "dist_based_score"])
def test_score_fusion_handles_unscored_nodes(mode: str):
    """Score-based fusion must not crash when a retriever yields unscored nodes."""
    retriever = QueryFusionRetriever(
        retrievers=[
            MockMixedRetriever(object_map={"sub_index": MockUnscoredRetriever()}),
            MockRetriever(),
        ],
        llm=MockLLM(),
        mode=mode,
        num_queries=1,
        use_async=False,
        similarity_top_k=10,
    )

    nodes = retriever.retrieve("test query")

    texts = [node.node.text for node in nodes]
    assert "unscored" in texts, f"Unscored node was dropped, got {texts}"
    assert all(isinstance(node.score, float) for node in nodes), (
        f"Expected every fused node to carry a float score, got "
        f"{[(n.node.text, n.score) for n in nodes]}"
    )


@pytest.mark.parametrize("mode", ["relative_score", "dist_based_score"])
def test_score_fusion_ranks_unscored_nodes_last(mode: str):
    """An unscored node is treated as 0.0 and must not outrank a scored one."""
    retriever = QueryFusionRetriever(
        retrievers=[
            MockMixedRetriever(object_map={"sub_index": MockUnscoredRetriever()})
        ],
        llm=MockLLM(),
        mode=mode,
        num_queries=1,
        use_async=False,
        similarity_top_k=10,
    )

    nodes = retriever.retrieve("test query")
    scores = {node.node.text: node.score for node in nodes}

    assert scores["scored"] > scores["unscored"], (
        f"Scored node should outrank unscored node, got {scores}"
    )
