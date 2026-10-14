import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class MockRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text="result"), score=1.0)]


class ReusingRetriever(BaseRetriever):
    def __init__(self, results):
        super().__init__()
        self.results = results

    def _retrieve(self, query_bundle: QueryBundle):
        return self.results


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


@pytest.mark.parametrize(
    "mode",
    [
        "reciprocal_rerank",
        "relative_score",
        "dist_based_score",
        "simple",
    ],
)
def test_fusion_does_not_mutate_retriever_results(mode):
    source = NodeWithScore(node=TextNode(text="result"), score=0.5)
    retriever = QueryFusionRetriever(
        retrievers=[ReusingRetriever([source])],
        mode=mode,
        num_queries=1,
        use_async=False,
    )

    result = retriever.retrieve("test query")

    assert result[0] is not source
    assert source.score == 0.5
