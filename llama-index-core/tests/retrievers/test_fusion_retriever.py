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


def test_relative_score_fusion_handles_none_scores():
    """_relative_score_fusion must not raise TypeError when node scores are None.

    NodeWithScore.score is Optional[float]; some retrievers legitimately return
    nodes without a score (score=None).  When the result set contains a mix of
    scored and unscored nodes the min/max bounds are computed with ``score or
    0.0`` (treating None as 0.0), but the per-node normalisation previously
    used ``node_with_score.score`` directly, raising::

        TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'

    After the fix, None scores are coerced to 0.0 before subtraction,
    consistent with the bounds computation.
    """
    retriever = QueryFusionRetriever.__new__(QueryFusionRetriever)
    retriever._retriever_weights = [1.0]
    retriever.num_queries = 1
    retriever.similarity_top_k = 10
    retriever.mode = FUSION_MODES.RELATIVE_SCORE

    node_with_score = NodeWithScore(node=TextNode(text="scored"), score=1.0)
    node_without_score = NodeWithScore(node=TextNode(text="unscored"), score=None)

    # Build the results dict the same way _run_sync_queries / _run_async_queries
    # would: keys are (query_str, retriever_idx), values are list of NodeWithScore.
    results = {
        ("test query", 0): [node_with_score, node_without_score],
    }

    # Must not raise TypeError
    reranked = retriever._relative_score_fusion(results)

    assert len(reranked) == 2
    # Scored node (1.0) should outrank the unscored node (treated as 0.0)
    assert reranked[0].node.text == "scored"
    assert reranked[1].node.text == "unscored"
    # All scores should be valid floats, not None
    for node in reranked:
        assert node.score is not None
        assert isinstance(node.score, float)


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
