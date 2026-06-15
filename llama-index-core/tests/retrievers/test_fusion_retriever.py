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


class FixedNodesRetriever(BaseRetriever):
    """Retriever that always returns a fixed list of nodes."""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        return self._nodes


def test_reciprocal_rerank_respects_retriever_weights():
    """`retriever_weights` must influence reciprocal_rerank fusion (issue #21444)."""
    retriever_0 = FixedNodesRetriever(
        [
            NodeWithScore(node=TextNode(text="node_A"), score=0.9),
            NodeWithScore(node=TextNode(text="node_B"), score=0.8),
        ]
    )
    retriever_1 = FixedNodesRetriever(
        [
            NodeWithScore(node=TextNode(text="node_C"), score=0.9),
            NodeWithScore(node=TextNode(text="node_D"), score=0.8),
        ]
    )

    retriever = QueryFusionRetriever(
        retrievers=[retriever_0, retriever_1],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        retriever_weights=[1.0, 0.0],
        num_queries=1,
        similarity_top_k=4,
        use_async=False,
    )

    results = retriever.retrieve("test query")
    contents = [n.node.get_content() for n in results]

    # retriever_1 has weight 0, so its nodes must not outrank retriever_0's.
    assert contents[:2] == ["node_A", "node_B"]

    # Nodes from the zero-weight retriever contribute nothing to the fused score.
    scores_by_content = {n.node.get_content(): n.score for n in results}
    assert scores_by_content["node_C"] == 0.0
    assert scores_by_content["node_D"] == 0.0
    assert scores_by_content["node_A"] > scores_by_content["node_B"] > 0.0


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
