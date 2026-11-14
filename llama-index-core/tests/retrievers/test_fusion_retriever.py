import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class MockRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text="result"), score=1.0)]


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


def test_relative_score_fusion_accepts_nodes_without_scores():
    """A node without a score must not crash relative-score fusion."""

    class MixedScoreRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: QueryBundle):
            return [
                NodeWithScore(node=TextNode(id_="no-score", text="a"), score=None),
                NodeWithScore(node=TextNode(id_="scored", text="b"), score=0.5),
            ]

    retriever = QueryFusionRetriever(
        retrievers=[MixedScoreRetriever()],
        llm=MockLLM(),
        mode=FUSION_MODES.RELATIVE_SCORE,
        num_queries=1,
        similarity_top_k=5,
    )

    nodes = retriever.retrieve("test query")

    assert [node.node.node_id for node in nodes] == ["scored", "no-score"]
    assert nodes[0].score == pytest.approx(1.0)
    assert nodes[1].score == pytest.approx(0.0)
