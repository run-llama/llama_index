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


def test_reciprocal_rerank_fusion_applies_retriever_weights():
    retriever = QueryFusionRetriever(
        retrievers=[MockRetriever(), MockRetriever()],
        llm=MockLLM(),
        mode=FUSION_MODES.RECIPROCAL_RANK,
        retriever_weights=[0.1, 0.9],
        use_async=False,
        num_queries=1,
    )
    lower_weight_node = NodeWithScore(
        node=TextNode(text="lower weight result"), score=1.0
    )
    higher_weight_node = NodeWithScore(
        node=TextNode(text="higher weight result"), score=1.0
    )

    results = retriever._reciprocal_rerank_fusion(
        {
            ("query", 0): [lower_weight_node],
            ("query", 1): [higher_weight_node],
        }
    )

    assert results[0].node.text == "higher weight result"
    assert (results[0].score or 0.0) > (results[1].score or 0.0)
