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


def _make_fusion(weights=None):
    return QueryFusionRetriever(
        retrievers=[MockRetriever(), MockRetriever()],
        llm=MockLLM(),
        mode="reciprocal_rerank",
        num_queries=1,
        retriever_weights=weights,
    )


def _node(text: str, score: float = 1.0) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text), score=score)


def test_reciprocal_rerank_applies_retriever_weights():
    """
    `_reciprocal_rerank_fusion` must scale each RRF contribution by the
    retriever's normalized weight.

    Regression test for the case where `retriever_weights` was silently
    ignored in `reciprocal_rerank` mode while it was honored in
    `relative_score` mode (issue #21444).
    """
    retriever = _make_fusion(weights=[3.0, 1.0])
    results = {("q", 0): [_node("a")], ("q", 1): [_node("b")]}

    fused = {
        nws.node.text: nws.score for nws in retriever._reciprocal_rerank_fusion(results)
    }

    # Normalized weights are [0.75, 0.25]; both docs are at rank 0 so the
    # only differentiator is the weight. Score ratio must equal weight ratio.
    assert fused["a"] == pytest.approx(0.75 / 60.0)
    assert fused["b"] == pytest.approx(0.25 / 60.0)
    assert fused["a"] == pytest.approx(fused["b"] * 3)


def test_reciprocal_rerank_equal_weights_preserve_ordering():
    """
    Equal weights must not change the *ordering* relative to the unweighted
    baseline — only the absolute score magnitude.

    Guards against accidentally introducing a weight-dependent tiebreaker.
    """
    results = {
        ("q", 0): [_node("doc_x"), _node("doc_y")],
        ("q", 1): [_node("doc_y"), _node("doc_x")],
    }

    fused = _make_fusion(weights=[1.0, 1.0])._reciprocal_rerank_fusion(results)
    scores = {nws.node.text: nws.score for nws in fused}

    # doc_x: rank0 from r0, rank1 from r1. doc_y: rank1 from r0, rank0 from r1.
    # Symmetric → scores must tie.
    assert scores["doc_x"] == pytest.approx(scores["doc_y"])


def test_reciprocal_rerank_zero_weight_suppresses_retriever():
    """
    A retriever with weight 0 must contribute 0 to fused scores — so a doc
    that only appears from that retriever ends up with score 0 and ranks
    behind any doc with non-zero weight contributions.
    """
    results = {
        ("q", 0): [_node("only_from_r0")],
        ("q", 1): [_node("only_from_r1")],
    }
    fused = _make_fusion(weights=[1.0, 0.0])._reciprocal_rerank_fusion(results)
    scores = {nws.node.text: nws.score for nws in fused}

    assert scores["only_from_r0"] > 0
    assert scores["only_from_r1"] == pytest.approx(0.0)
    # Returned in score-desc order.
    assert fused[0].node.text == "only_from_r0"


def test_reciprocal_rerank_weight_promotes_relevant_doc():
    """
    Semantic / impact test: when one retriever ranks the relevant doc first
    and the other ranks an irrelevant doc first, weighting the *good*
    retriever higher must make the relevant doc win, and weighting the
    *bad* retriever higher must reverse the outcome.
    """

    # r0 is the "good" retriever: relevant doc at rank 0.
    # r1 is the "bad" retriever: irrelevant doc at rank 0.
    # Note: `_reciprocal_rerank_fusion` mutates `node_with_score.score`
    # on its inputs, so build fresh node lists for each call.
    def fresh_inputs():
        return {
            ("q", 0): [_node("relevant"), _node("irrelevant")],
            ("q", 1): [_node("irrelevant"), _node("relevant")],
        }

    favor_good = _make_fusion(weights=[5.0, 1.0])._reciprocal_rerank_fusion(
        fresh_inputs()
    )
    favor_bad = _make_fusion(weights=[1.0, 5.0])._reciprocal_rerank_fusion(
        fresh_inputs()
    )

    assert favor_good[0].node.text == "relevant"
    assert favor_bad[0].node.text == "irrelevant"
