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


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("rewrites", ["original\nrewrite\nrewrite", "rewrite"])
@pytest.mark.parametrize("mode", ["relative_score", "dist_based_score"])
async def test_fusion_normalizes_by_distinct_queries(
    use_async: bool, rewrites: str, mode: str
) -> None:
    queries = []

    class RewritingLLM(MockLLM):
        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            return CompletionResponse(text=rewrites)

        async def acomplete(self, prompt: str, formatted: bool = False, **kwargs):
            return CompletionResponse(text=rewrites)

    class RecordingRetriever(MockRetriever):
        def _retrieve(self, query_bundle: QueryBundle):
            queries.append(query_bundle.query_str)
            return super()._retrieve(query_bundle)

    retriever = QueryFusionRetriever(
        retrievers=[RecordingRetriever()],
        llm=RewritingLLM(),
        mode=FUSION_MODES(mode),
        num_queries=4,
        use_async=False,
    )
    if use_async:
        results = await retriever.aretrieve("original")
    else:
        results = retriever.retrieve("original")

    assert queries == ["original", "rewrite"]
    assert [result.score for result in results] == [1.0]
