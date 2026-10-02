import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class MockRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text="result"), score=1.0)]


class CachedRetriever(BaseRetriever):
    def __init__(self, copy_results: bool = False):
        super().__init__()
        self.copy_results = copy_results
        shared = NodeWithScore(
            node=TextNode(text="relevant", id_="relevant"), score=0.9
        )
        self.results = {
            "original": [
                shared,
                NodeWithScore(node=TextNode(text="low", id_="low"), score=0.1),
            ],
            "alternate": [
                shared,
                NodeWithScore(node=TextNode(text="other", id_="other"), score=0.8),
            ],
        }

    def _retrieve(self, query_bundle: QueryBundle):
        result = self.results[query_bundle.query_str]
        return [node.model_copy() for node in result] if self.copy_results else result


class AlternateQueryLLM(MockLLM):
    def complete(self, prompt: str, formatted: bool = False, **kwargs):
        return CompletionResponse(text="alternate")

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs):
        return CompletionResponse(text="alternate")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["relative_score", "dist_based_score"])
@pytest.mark.parametrize("async_api", [False, True])
async def test_score_fusion_preserves_shared_cached_results(mode, async_api):
    outputs = []
    for copy_results in [True, False]:
        base = CachedRetriever(copy_results=copy_results)
        fusion = QueryFusionRetriever(
            retrievers=[base],
            llm=AlternateQueryLLM(),
            num_queries=2,
            mode=mode,
            use_async=False,
            similarity_top_k=3,
        )
        result = (
            await fusion.aretrieve("original")
            if async_api
            else fusion.retrieve("original")
        )

        assert result[0].node_id == "relevant"
        outputs.append({node.node_id: node.score for node in result})
        assert [node.score for node in base.results["original"]] == [0.9, 0.1]
        assert [node.score for node in base.results["alternate"]] == [0.9, 0.8]

    assert outputs[1] == pytest.approx(outputs[0])


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
