import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class MockRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text="result"), score=1.0)]


class AlternateQueryLLM(MockLLM):
    def complete(self, prompt: str, formatted: bool = False, **kwargs):
        return CompletionResponse(text="alternate")

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs):
        return CompletionResponse(text="alternate")


class CachedRetriever(BaseRetriever):
    def __init__(self):
        super().__init__()
        self.shared = NodeWithScore(
            node=TextNode(text="relevant", id_="relevant"), score=0.9
        )
        self.results = {
            "original": [
                self.shared,
                NodeWithScore(node=TextNode(text="low", id_="low"), score=0.1),
            ],
            "alternate": [
                self.shared,
                NodeWithScore(node=TextNode(text="other", id_="other"), score=0.8),
            ],
        }

    def _retrieve(self, query_bundle: QueryBundle):
        return self.results[query_bundle.query_str]


def get_reciprocal_rank_retriever(base_retriever: BaseRetriever):
    return QueryFusionRetriever(
        retrievers=[base_retriever],
        llm=AlternateQueryLLM(),
        num_queries=2,
        mode="reciprocal_rerank",
        use_async=False,
        similarity_top_k=3,
    )


def assert_cached_scores_are_unchanged(base_retriever: CachedRetriever):
    assert [node.score for node in base_retriever.results["original"]] == [0.9, 0.1]
    assert [node.score for node in base_retriever.results["alternate"]] == [0.9, 0.8]


def test_reciprocal_rank_fusion_does_not_mutate_retriever_results():
    base_retriever = CachedRetriever()

    results = get_reciprocal_rank_retriever(base_retriever).retrieve("original")

    assert [node.node_id for node in results] == ["relevant", "low", "other"]
    assert results[0].node is base_retriever.shared.node
    assert results[0] is not base_retriever.shared
    assert_cached_scores_are_unchanged(base_retriever)


@pytest.mark.asyncio
async def test_async_reciprocal_rank_fusion_does_not_mutate_retriever_results():
    base_retriever = CachedRetriever()

    results = await get_reciprocal_rank_retriever(base_retriever).aretrieve("original")

    assert [node.node_id for node in results] == ["relevant", "low", "other"]
    assert results[0].node is base_retriever.shared.node
    assert results[0] is not base_retriever.shared
    assert_cached_scores_are_unchanged(base_retriever)


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
