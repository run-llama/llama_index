import math
from typing import Any, ClassVar, Dict, List, Optional

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator
from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)
from llama_index.core.evaluation.retrieval.segmented_evaluator import (
    SegmentedRetrievalEvalResult,
    SegmentedRetrieverEvaluator,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

METRIC_NAMES = ["hit_rate", "mrr"]


class MockRetriever(BaseRetriever):
    """Retriever returning a fixed node list, counting how often it is called."""

    def __init__(self, nodes: List[NodeWithScore]) -> None:
        self._nodes = nodes
        self.num_calls = 0
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        self.num_calls += 1
        return list(self._nodes)


class DropSegmentPostprocessor(BaseNodePostprocessor):
    """Drops every node belonging to a given segment."""

    segment: str

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        return [node for node in nodes if node.metadata["segment"] != self.segment]


class RecordingMetric(BaseRetrievalMetric):
    """Records the arguments it receives, to pin down the calling convention."""

    metric_name: ClassVar[str] = "recording"
    calls: List[Dict[str, Any]] = Field(default_factory=list)

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        self.calls.append(
            {
                "query": query,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "expected_texts": expected_texts,
                "retrieved_texts": retrieved_texts,
            }
        )
        return RetrievalMetricResult(score=0.0)


def make_node(node_id: str, segment: str) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            id_=node_id, text=f"text of {node_id}", metadata={"segment": segment}
        ),
        score=1.0,
    )


def segment_fn(node: NodeWithScore) -> str:
    return str(node.metadata["segment"])


def make_evaluator(
    nodes: List[NodeWithScore],
    segments: Optional[List[str]] = None,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    metric_names: Optional[List[str]] = None,
    expected_segments: Optional[Dict[str, str]] = None,
) -> SegmentedRetrieverEvaluator:
    metric_names = metric_names or METRIC_NAMES
    retriever = MockRetriever(nodes)
    base_evaluator = RetrieverEvaluator.from_metric_names(
        metric_names, retriever=retriever, node_postprocessors=node_postprocessors
    )
    return SegmentedRetrieverEvaluator.from_metric_names(
        metric_names,
        base_evaluator=base_evaluator,
        segment_fn=segment_fn,
        segments=segments,
        expected_segments=expected_segments,
    )


# [n1(A), n2(B), n3(A)] with n3 relevant: within segment A, n3 sits at local
# rank 2, so mrr_A is 1/2 rather than the global 1/3.
INTERLEAVED_NODES = [
    make_node("n1", "A"),
    make_node("n2", "B"),
    make_node("n3", "A"),
]


@pytest.mark.asyncio
async def test_mrr_uses_segment_local_ranks() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)

    result = await evaluator.aevaluate(query="q", expected_ids=["n3"])

    segment_vals = result.segment_metric_vals_dict
    assert segment_vals["A"]["mrr"] == pytest.approx(1 / 2)
    assert segment_vals["A"]["hit_rate"] == pytest.approx(1.0)
    assert segment_vals["B"]["mrr"] == pytest.approx(0.0)
    assert segment_vals["B"]["hit_rate"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_overall_metrics_match_plain_evaluator() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)
    plain_evaluator = RetrieverEvaluator.from_metric_names(
        METRIC_NAMES, retriever=MockRetriever(INTERLEAVED_NODES)
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["n3"])
    plain_result = await plain_evaluator.aevaluate(query="q", expected_ids=["n3"])

    assert result.metric_vals_dict == plain_result.metric_vals_dict
    assert result.metric_vals_dict["mrr"] == pytest.approx(1 / 3)
    assert result.retrieved_ids == plain_result.retrieved_ids


@pytest.mark.asyncio
async def test_empty_retrieval_raises_like_plain_evaluator() -> None:
    # The wrapper must not soften the overall behaviour of what it wraps: the
    # per-segment zero-fill applies to segments only.
    evaluator = make_evaluator([])
    plain_evaluator = RetrieverEvaluator.from_metric_names(
        METRIC_NAMES, retriever=MockRetriever([])
    )

    with pytest.raises(ValueError, match="must be provided"):
        await plain_evaluator.aevaluate(query="q", expected_ids=["n3"])

    with pytest.raises(ValueError, match="must be provided"):
        await evaluator.aevaluate(query="q", expected_ids=["n3"])


@pytest.mark.asyncio
async def test_metric_arguments_are_passed_in_the_expected_order() -> None:
    # The five compute() arguments are passed positionally and are all of
    # similar shape, so a swap would go unnoticed by hit_rate and mrr.
    metric = RecordingMetric()
    evaluator = SegmentedRetrieverEvaluator(
        metrics=[metric],
        base_evaluator=RetrieverEvaluator(
            metrics=[], retriever=MockRetriever(INTERLEAVED_NODES)
        ),
        segment_fn=segment_fn,
    )

    await evaluator.aevaluate(
        query="q", expected_ids=["n3"], expected_texts=["text of n3"]
    )

    overall_call, segment_a_call, segment_b_call = metric.calls
    assert overall_call == {
        "query": "q",
        "expected_ids": ["n3"],
        "retrieved_ids": ["n1", "n2", "n3"],
        "expected_texts": ["text of n3"],
        "retrieved_texts": ["text of n1", "text of n2", "text of n3"],
    }
    # Segments narrow the retrieved side only; the expected side stays global.
    assert segment_a_call["retrieved_ids"] == ["n1", "n3"]
    assert segment_a_call["retrieved_texts"] == ["text of n1", "text of n3"]
    assert segment_a_call["expected_ids"] == ["n3"]
    assert segment_a_call["expected_texts"] == ["text of n3"]
    assert segment_b_call["retrieved_ids"] == ["n2"]
    assert segment_b_call["retrieved_texts"] == ["text of n2"]


@pytest.mark.asyncio
async def test_metrics_other_than_hit_rate_and_mrr() -> None:
    evaluator = make_evaluator(
        INTERLEAVED_NODES, metric_names=["precision", "recall", "ap"]
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["n3"])

    assert result.metric_vals_dict["precision"] == pytest.approx(1 / 3)
    segment_vals = result.segment_metric_vals_dict
    assert segment_vals["A"]["precision"] == pytest.approx(1 / 2)
    assert segment_vals["B"]["precision"] == pytest.approx(0.0)
    assert segment_vals["A"]["ap"] == pytest.approx(1 / 2)
    # recall and ap normalize against the *global* expected ids, which is why
    # the docstring flags them as being of limited use per segment.
    assert segment_vals["A"]["recall"] == pytest.approx(1.0)
    assert segment_vals["B"]["recall"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_declared_segment_without_nodes_scores_zero() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES, segments=["A", "B", "C"])

    result = await evaluator.aevaluate(query="q", expected_ids=["n3"])

    # "C" never appears in the retrieved nodes: the metrics reject empty id
    # lists, so this must not raise and must report 0.0 instead.
    assert list(result.segment_metric_dict) == ["A", "B", "C"]
    assert result.segment_retrieved_ids["C"] == []
    assert result.segment_metric_vals_dict["C"] == {"hit_rate": 0.0, "mrr": 0.0}


@pytest.mark.asyncio
async def test_undeclared_segment_label_raises() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES, segments=["A"])

    with pytest.raises(ValueError, match="'B'"):
        await evaluator.aevaluate(query="q", expected_ids=["n3"])


@pytest.mark.asyncio
async def test_without_declared_segments_only_observed_are_reported() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)

    result = await evaluator.aevaluate(query="q", expected_ids=["n3"])

    assert set(result.segment_metric_dict) == {"A", "B"}
    assert result.segment_retrieved_ids == {"A": ["n1", "n3"], "B": ["n2"]}


@pytest.mark.asyncio
async def test_node_postprocessors_are_applied() -> None:
    evaluator = make_evaluator(
        INTERLEAVED_NODES,
        node_postprocessors=[DropSegmentPostprocessor(segment="B")],
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["n3"])

    assert set(result.segment_metric_dict) == {"A"}
    assert result.retrieved_ids == ["n1", "n3"]


@pytest.mark.asyncio
async def test_retrieval_runs_exactly_once() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)
    retriever = evaluator.base_evaluator.retriever
    assert isinstance(retriever, MockRetriever)

    await evaluator.aevaluate(query="q", expected_ids=["n3"])

    assert retriever.num_calls == 1


def test_sync_evaluate_returns_segmented_result() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)

    result = evaluator.evaluate(query="q", expected_ids=["n3"])

    assert isinstance(result, SegmentedRetrievalEvalResult)
    assert result.segment_metric_vals_dict["A"]["mrr"] == pytest.approx(1 / 2)


@pytest.mark.asyncio
async def test_ids_and_texts_delegate_to_wrapped_evaluator() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)

    ids, texts = await evaluator._aget_retrieved_ids_and_texts("q")

    assert ids == ["n1", "n2", "n3"]
    assert texts == ["text of n1", "text of n2", "text of n3"]


def test_str_includes_segment_breakdown() -> None:
    evaluator = make_evaluator(INTERLEAVED_NODES)

    output = str(evaluator.evaluate(query="q", expected_ids=["n3"]))

    assert "Query: q" in output
    assert "Segments:" in output
    assert "A:" in output
    assert "B:" in output


# Expected-side segments: the whole corpus is labelled, as a user would do once
# from the nodes the index was built from.
RANKED_METRIC_NAMES = ["hit_rate", "mrr", "recall", "ap", "ndcg"]
CORPUS_SEGMENTS = {
    "p1": "prose",
    "p2": "prose",
    "p3": "prose",
    "t1": "table",
    "t2": "table",
}

# [p1(prose), t1(table), p2(prose), t2(table)]: every segment has its relevant
# node at local rank 2, while the global ranks are 3 and 4.
TABLE_PROSE_NODES = [
    make_node("p1", "prose"),
    make_node("t1", "table"),
    make_node("p2", "prose"),
    make_node("t2", "table"),
]

# 1 / log2(3): DCG of a single hit at rank 2, against an IDCG of 1.
NDCG_HIT_AT_RANK_2 = 1 / math.log2(3)


@pytest.mark.asyncio
async def test_relevant_segment_without_retrieved_nodes_scores_zero() -> None:
    # The answer lives in a table, but only prose is retrieved: that is a
    # table failure and must show up as such.
    evaluator = make_evaluator(
        [make_node("p1", "prose"), make_node("p2", "prose")],
        metric_names=RANKED_METRIC_NAMES,
        expected_segments=CORPUS_SEGMENTS,
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["t1"])

    assert result.segment_metric_vals_dict["table"] == dict.fromkeys(
        RANKED_METRIC_NAMES, 0.0
    )
    assert result.segment_retrieved_ids["table"] == []
    assert result.not_applicable_segments == ["prose"]


@pytest.mark.asyncio
async def test_segment_without_relevant_expected_ids_is_not_applicable() -> None:
    evaluator = make_evaluator(
        TABLE_PROSE_NODES,
        segments=["prose", "table", "image"],
        expected_segments=CORPUS_SEGMENTS,
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["p2"])

    # Neither the retrieved-but-irrelevant table nor the absent image segment
    # may score 0.0; both are reported as not applicable instead.
    assert list(result.segment_metric_dict) == ["prose"]
    assert result.not_applicable_segments == ["table", "image"]
    assert result.segment_retrieved_ids == {
        "prose": ["p1", "p2"],
        "table": ["t1", "t2"],
        "image": [],
    }
    assert "table: n/a" in str(result)


@pytest.mark.asyncio
async def test_interleaved_segments_use_segment_expected_ids() -> None:
    evaluator = make_evaluator(
        TABLE_PROSE_NODES,
        metric_names=RANKED_METRIC_NAMES,
        expected_segments=CORPUS_SEGMENTS,
    )
    plain_evaluator = RetrieverEvaluator.from_metric_names(
        RANKED_METRIC_NAMES, retriever=MockRetriever(TABLE_PROSE_NODES)
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["t2", "p2"])
    plain_result = await plain_evaluator.aevaluate(query="q", expected_ids=["t2", "p2"])

    assert result.metric_vals_dict == plain_result.metric_vals_dict
    assert result.not_applicable_segments == []
    for segment in ("prose", "table"):
        segment_vals = result.segment_metric_vals_dict[segment]
        assert segment_vals["hit_rate"] == pytest.approx(1.0)
        assert segment_vals["mrr"] == pytest.approx(1 / 2)
        # Normalized against the one relevant id of the segment, not both.
        assert segment_vals["recall"] == pytest.approx(1.0)
        assert segment_vals["ap"] == pytest.approx(1 / 2)
        assert segment_vals["ndcg"] == pytest.approx(NDCG_HIT_AT_RANK_2)


@pytest.mark.asyncio
async def test_without_expected_segments_behaviour_is_unchanged() -> None:
    evaluator = make_evaluator(
        TABLE_PROSE_NODES,
        segments=["prose", "table", "image"],
        metric_names=RANKED_METRIC_NAMES,
    )

    result = await evaluator.aevaluate(query="q", expected_ids=["t2", "p2"])

    # Every segment is still scored against the global expected ids, and a
    # declared segment without retrieved nodes still scores 0.0.
    assert result.not_applicable_segments == []
    assert list(result.segment_metric_dict) == ["prose", "table", "image"]
    prose_vals = result.segment_metric_vals_dict["prose"]
    assert prose_vals["mrr"] == pytest.approx(1 / 2)
    assert prose_vals["recall"] == pytest.approx(1 / 2)
    assert prose_vals["ap"] == pytest.approx(1 / 4)
    assert prose_vals["ndcg"] == pytest.approx(
        NDCG_HIT_AT_RANK_2 / (1 + NDCG_HIT_AT_RANK_2)
    )
    assert result.segment_metric_vals_dict["image"] == dict.fromkeys(
        RANKED_METRIC_NAMES, 0.0
    )


@pytest.mark.asyncio
async def test_expected_segments_narrow_expected_ids_and_texts() -> None:
    metric = RecordingMetric()
    evaluator = SegmentedRetrieverEvaluator(
        metrics=[metric],
        base_evaluator=RetrieverEvaluator(
            metrics=[], retriever=MockRetriever(TABLE_PROSE_NODES)
        ),
        segment_fn=segment_fn,
        segments=["prose", "table", "image"],
        expected_segments=CORPUS_SEGMENTS,
    )

    await evaluator.aevaluate(
        query="q",
        expected_ids=["t2", "p2"],
        expected_texts=["text of t2", "text of p2"],
    )

    # No call for the not-applicable "image" segment.
    overall_call, prose_call, table_call = metric.calls
    assert overall_call["expected_ids"] == ["t2", "p2"]
    assert prose_call["expected_ids"] == ["p2"]
    assert prose_call["expected_texts"] == ["text of p2"]
    assert prose_call["retrieved_ids"] == ["p1", "p2"]
    assert table_call["expected_ids"] == ["t2"]
    assert table_call["expected_texts"] == ["text of t2"]
    assert table_call["retrieved_ids"] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_non_string_segment_label_raises() -> None:
    evaluator = SegmentedRetrieverEvaluator.from_metric_names(
        METRIC_NAMES,
        base_evaluator=RetrieverEvaluator.from_metric_names(
            METRIC_NAMES, retriever=MockRetriever(INTERLEAVED_NODES)
        ),
        segment_fn=lambda node: None,
    )

    with pytest.raises(ValueError, match="segment_fn must return a str"):
        await evaluator.aevaluate(query="q", expected_ids=["n3"])


@pytest.mark.asyncio
async def test_expected_id_without_segment_raises() -> None:
    evaluator = make_evaluator(TABLE_PROSE_NODES, expected_segments=CORPUS_SEGMENTS)

    with pytest.raises(ValueError, match="'x1'"):
        await evaluator.aevaluate(query="q", expected_ids=["p2", "x1"])


@pytest.mark.asyncio
async def test_conflicting_segment_labels_raise() -> None:
    evaluator = make_evaluator(
        TABLE_PROSE_NODES,
        expected_segments={**CORPUS_SEGMENTS, "t1": "prose"},
    )

    with pytest.raises(ValueError, match="'t1'"):
        await evaluator.aevaluate(query="q", expected_ids=["p2"])


@pytest.mark.asyncio
async def test_undeclared_expected_segment_raises() -> None:
    evaluator = make_evaluator(
        [make_node("p1", "prose")],
        segments=["prose"],
        expected_segments=CORPUS_SEGMENTS,
    )

    with pytest.raises(ValueError, match="'table' from expected_segments"):
        await evaluator.aevaluate(query="q", expected_ids=["t1"])
