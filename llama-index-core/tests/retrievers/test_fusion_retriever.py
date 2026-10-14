import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
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


class QueryLLM(MockLLM):
    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(text="alternate")

    async def acomplete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(text="alternate")


class PerQueryCacheRetriever(BaseRetriever):
    """Per-query cache: the same node hash arrives via distinct wrappers."""

    def __init__(self):
        super().__init__()
        self.node = TextNode(text="relevant", id_="relevant")
        self.results = {
            "original": [
                NodeWithScore(node=self.node, score=0.4),
                NodeWithScore(node=TextNode(text="low", id_="low"), score=0.1),
            ],
            "alternate": [
                NodeWithScore(node=self.node, score=0.9),
                NodeWithScore(node=TextNode(text="other", id_="other"), score=0.8),
            ],
        }

    def _retrieve(self, query_bundle: QueryBundle):
        return self.results[query_bundle.query_str]


class SharedWrapperRetriever(BaseRetriever):
    """Node-level cache: one wrapper object shared across queries (#23332)."""

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


def _simple_fusion_retriever(base):
    return QueryFusionRetriever(
        [base],
        llm=QueryLLM(),
        num_queries=2,
        mode="simple",
        use_async=False,
        similarity_top_k=3,
    )


def test_simple_fusion_preserves_per_query_cached_scores():
    # SIMPLE half of #23351: the dedup max-write used to put the max score of
    # another query into the wrapper owned by "original"'s cached results.
    base = PerQueryCacheRetriever()
    fusion = _simple_fusion_retriever(base)

    result = fusion.retrieve("original")

    assert [(n.node_id, n.score) for n in result] == [
        ("relevant", 0.9),
        ("other", 0.8),
        ("low", 0.1),
    ]
    assert [(n.node_id, n.score) for n in base.results["original"]] == [
        ("relevant", 0.4),
        ("low", 0.1),
    ]
    assert [(n.node_id, n.score) for n in base.results["alternate"]] == [
        ("relevant", 0.9),
        ("other", 0.8),
    ]


@pytest.mark.asyncio
async def test_simple_fusion_async_preserves_per_query_cached_scores():
    base = PerQueryCacheRetriever()
    fusion = _simple_fusion_retriever(base)

    result = await fusion.aretrieve("original")

    assert [(n.node_id, n.score) for n in result] == [
        ("relevant", 0.9),
        ("other", 0.8),
        ("low", 0.1),
    ]
    assert [(n.node_id, n.score) for n in base.results["original"]] == [
        ("relevant", 0.4),
        ("low", 0.1),
    ]


def test_simple_fusion_preserves_shared_wrapper_cache():
    # The shared-wrapper repro from #23332 is a no-op for the dedup write
    # (max(0.9, 0.9) is identity), but the output must not alias the
    # retriever's cached wrapper either.
    base = SharedWrapperRetriever()
    fusion = _simple_fusion_retriever(base)

    result = fusion.retrieve("original")

    assert [(n.node_id, n.score) for n in result] == [
        ("relevant", 0.9),
        ("other", 0.8),
        ("low", 0.1),
    ]
    assert base.shared.score == 0.9
    assert base.results["original"][1].score == 0.1
    assert base.results["alternate"][1].score == 0.8


def test_simple_fusion_output_does_not_alias_retriever_wrappers():
    # Mutating the fused output must never reach the retriever's cache.
    base = PerQueryCacheRetriever()
    fusion = _simple_fusion_retriever(base)

    result = fusion.retrieve("original")
    result[0].score = 999.0

    assert base.results["original"][0].score == 0.4
    assert base.results["alternate"][0].score == 0.9
