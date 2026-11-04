"""Segmented retrieval evaluator."""

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from llama_index.core.bridge.pydantic import Field
from llama_index.core.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalMode,
    RetrievalEvalResult,
)
from llama_index.core.evaluation.retrieval.evaluator import (
    RetrieverEvaluator,
    _nodes_to_ids_and_texts,
)
from llama_index.core.evaluation.retrieval.metrics_base import RetrievalMetricResult
from llama_index.core.schema import NodeWithScore


class SegmentedRetrievalEvalResult(RetrievalEvalResult):
    """
    Retrieval eval result broken down by segment.

    Extends `RetrievalEvalResult` with per-segment metrics, so the overall
    scores stay available in `metric_dict` and existing helpers such as
    `get_retrieval_results_df` keep working unchanged.

    Every segment of a result appears in exactly one of `segment_metric_dict`
    and `not_applicable_segments`, so aggregating over `segment_metric_dict`
    skips not-applicable segments automatically.

    Attributes:
        segment_metric_dict (Dict[str, Dict[str, RetrievalMetricResult]]): \
            Segment name to metric name to metric result
        segment_retrieved_ids (Dict[str, List[str]]): Segment name to the ids \
            retrieved for that segment, in retrieval order. Includes \
            not-applicable segments.
        not_applicable_segments (List[str]): Segments without any relevant \
            expected ids for this query, which therefore have no metrics. Only \
            ever non-empty when the evaluator has `expected_segments` set.

    """

    segment_metric_dict: Dict[str, Dict[str, RetrievalMetricResult]] = Field(
        ..., description="Metric dictionary per segment"
    )
    segment_retrieved_ids: Dict[str, List[str]] = Field(
        ..., description="Retrieved ids per segment, in retrieval order"
    )
    not_applicable_segments: List[str] = Field(
        default_factory=list,
        description="Segments without relevant expected ids, scored by no metric",
    )

    @property
    def segment_metric_vals_dict(self) -> Dict[str, Dict[str, float]]:
        """Dictionary of metric values per segment."""
        return {
            segment: {k: v.score for k, v in metric_dict.items()}
            for segment, metric_dict in self.segment_metric_dict.items()
        }

    def __str__(self) -> str:
        """String representation."""
        segment_lines = "".join(
            f"  {segment}: {metric_vals!s}\n"
            for segment, metric_vals in self.segment_metric_vals_dict.items()
        )
        segment_lines += "".join(
            f"  {segment}: n/a\n" for segment in self.not_applicable_segments
        )
        return (
            f"Query: {self.query}\n"
            f"Metrics: {self.metric_vals_dict!s}\n"
            f"Segments:\n{segment_lines}"
        )


class SegmentedRetrieverEvaluator(BaseRetrievalEvaluator):
    """
    Segmented retriever evaluator.

    Wraps a `RetrieverEvaluator` and additionally computes every configured
    metric per segment, where `segment_fn` assigns a segment label to each
    retrieved node. Useful to find out *where* a retriever underperforms, e.g.
    per source format, language or document collection.

    Ranks are segment-local: the retrieved list is filtered down to a segment
    and ranks are counted from 1 again. Rank-aware metrics such as `mrr`, `ap`
    and `ndcg` therefore see the rank *within the segment*, not the rank the
    user actually sees in the full result list. A node at overall position 3
    that is second within its segment contributes an MRR of 1/2, not 1/3.

    Expected ids are labelled with `expected_segments`, a mapping from expected
    id to segment. Each segment is then scored against only its own expected
    ids, which gives three outcomes per segment:

    - relevant expected ids and retrieved hits: metrics as usual,
    - relevant expected ids but no retrieved nodes in the segment: every metric
      scores 0.0, e.g. the answer lives in a table but only prose came back,
    - no relevant expected ids: the segment is not applicable for this query.
      It gets no metrics and is listed in `not_applicable_segments` instead,
      so averages across queries are not dragged down by it.

    Without `expected_segments`, only retrieved nodes carry a label. There is
    then no way to tell whether a segment contained anything relevant in the
    first place: a `hit_rate` of 0.0 means "nothing relevant was found here",
    not necessarily "the retriever failed here", and a relevant segment that
    was never retrieved does not show up at all. Every segment is scored
    against the *global* expected ids, so `recall`, `ndcg` and `ap` are of
    limited use per segment, and a declared segment without retrieved nodes
    scores 0.0. Use `segment_retrieved_ids` to skip such segments when
    aggregating: an empty list means the segment contributed no nodes at all.

    NOTE: `segment_fn` is a callable and is excluded from serialization, so a
    `model_dump()` of this evaluator cannot be used to reconstruct it.

    NOTE: `mode` is accepted for interface compatibility but ignored, same as
    in `RetrieverEvaluator`: retrieval always runs as text retrieval.

    Unlike per-segment metrics, the overall metrics keep the behaviour of the
    wrapped evaluator: an empty retrieval result raises rather than scoring 0.0.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        base_evaluator (RetrieverEvaluator): The retriever evaluator to wrap
        segment_fn (Callable[[NodeWithScore], str]): Assigns a segment label \
            to a retrieved node. Must return a str.
        segments (Optional[List[str]]): Fixed set of segment labels. When set, \
            every result carries exactly these segments, which makes results \
            comparable across queries. A label returned by `segment_fn` or \
            found in `expected_segments` that is not in this list raises a \
            ValueError.
        expected_segments (Optional[Dict[str, str]]): Maps expected ids to \
            their segment label, usually built once for the whole corpus. \
            Every expected id of a query must be in it, and a retrieved node \
            listed in it must get the same label from `segment_fn`.

    Examples:
        ```python
        from llama_index.core.evaluation import (
            RetrieverEvaluator,
            SegmentedRetrieverEvaluator,
        )


        def segment_fn(node):
            return node.metadata["format"]  # e.g. "table" or "prose"


        # Label the corpus once, using the same labels as `segment_fn`.
        expected_segments = {
            node.node_id: node.metadata["format"] for node in nodes
        }

        evaluator = SegmentedRetrieverEvaluator.from_metric_names(
            ["hit_rate", "mrr", "recall"],
            base_evaluator=RetrieverEvaluator.from_metric_names(
                ["hit_rate", "mrr", "recall"], retriever=retriever
            ),
            segment_fn=segment_fn,
            segments=["table", "prose"],
            expected_segments=expected_segments,
        )

        result = evaluator.evaluate(query="...", expected_ids=["table_node_id"])
        result.segment_metric_vals_dict  # {"table": {...}}
        result.not_applicable_segments  # ["prose"]
        ```

    """

    base_evaluator: RetrieverEvaluator = Field(
        ..., description="Retriever evaluator to wrap"
    )
    segment_fn: Callable[[NodeWithScore], str] = Field(
        ...,
        description="Assigns a segment label to a retrieved node",
        exclude=True,
    )
    segments: Optional[List[str]] = Field(
        default=None,
        description=(
            "Fixed set of segment labels, for stable keys across queries. "
            "Defaults to whichever segments are observed per query."
        ),
    )
    expected_segments: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Maps expected ids to their segment label. When set, each segment "
            "is scored against its own expected ids only."
        ),
    )

    async def _aget_retrieved_ids_and_texts(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        """Get retrieved ids and texts from the wrapped evaluator."""
        return _nodes_to_ids_and_texts(
            await self.base_evaluator.aget_retrieved_nodes(query)
        )

    def _check_declared(self, segment: str, source: str) -> None:
        """Raise if `segments` is declared and does not contain `segment`."""
        if self.segments is not None and segment not in self.segments:
            raise ValueError(
                f"Segment {segment!r} from {source} is not in the declared "
                f"segments {self.segments!r}."
            )

    def _label_node(self, node: NodeWithScore) -> str:
        """Label a retrieved node with `segment_fn`, validating the result."""
        segment = self.segment_fn(node)
        if not isinstance(segment, str):
            raise ValueError(
                f"segment_fn must return a str, got {type(segment).__name__} "
                f"for node {node.node.node_id!r}."
            )
        self._check_declared(segment, "segment_fn")

        if self.expected_segments is not None:
            expected_segment = self.expected_segments.get(node.node.node_id)
            if expected_segment is not None and expected_segment != segment:
                raise ValueError(
                    f"segment_fn labels node {node.node.node_id!r} as "
                    f"{segment!r}, but expected_segments labels it as "
                    f"{expected_segment!r}."
                )

        return segment

    def _group_nodes_by_segment(
        self, nodes: List[NodeWithScore]
    ) -> Dict[str, List[NodeWithScore]]:
        """Group retrieved nodes by segment, preserving retrieval order."""
        grouped: Dict[str, List[NodeWithScore]] = {}
        if self.segments is not None:
            # Seed the declared segments so they are always present and ordered
            grouped = {segment: [] for segment in self.segments}

        for node in nodes:
            grouped.setdefault(self._label_node(node), []).append(node)

        return grouped

    def _group_expected_by_segment(
        self,
        expected_segments: Dict[str, str],
        expected_ids: List[str],
        expected_texts: Optional[List[str]],
    ) -> Dict[str, Tuple[List[str], Optional[List[str]]]]:
        """
        Group expected ids, and expected texts if given, by segment.

        Expected texts are matched to expected ids by position.
        """
        if expected_texts is not None and len(expected_texts) != len(expected_ids):
            raise ValueError(
                "expected_texts must have the same length as expected_ids to be "
                "split by segment."
            )

        grouped: Dict[str, Tuple[List[str], Optional[List[str]]]] = {}
        for i, expected_id in enumerate(expected_ids):
            if expected_id not in expected_segments:
                raise ValueError(
                    f"Expected id {expected_id!r} has no segment in expected_segments."
                )
            segment = expected_segments[expected_id]
            self._check_declared(segment, "expected_segments")

            segment_ids, segment_texts = grouped.setdefault(
                segment, ([], None if expected_texts is None else [])
            )
            segment_ids.append(expected_id)
            if segment_texts is not None and expected_texts is not None:
                segment_texts.append(expected_texts[i])

        return grouped

    def _compute_metrics(
        self,
        query: str,
        expected_ids: List[str],
        retrieved_ids: List[str],
        expected_texts: Optional[List[str]],
        retrieved_texts: List[str],
    ) -> Dict[str, RetrievalMetricResult]:
        """
        Compute every configured metric for one set of retrieved ids.

        Mirrors `BaseRetrievalEvaluator.aevaluate`, including its error
        behaviour on empty retrieval results.
        """
        return {
            metric.metric_name: metric.compute(
                query, expected_ids, retrieved_ids, expected_texts, retrieved_texts
            )
            for metric in self.metrics
        }

    def _compute_segment_metrics(
        self,
        query: str,
        expected_ids: List[str],
        retrieved_ids: List[str],
        expected_texts: Optional[List[str]],
        retrieved_texts: List[str],
    ) -> Dict[str, RetrievalMetricResult]:
        """
        Compute every configured metric for a single segment.

        Unlike the overall metrics, a segment without retrieved nodes scores
        0.0 rather than raising: metrics such as HitRate and MRR reject empty
        retrieval results, but an empty segment is a meaningful outcome.
        """
        if not retrieved_ids:
            return {
                metric.metric_name: RetrievalMetricResult(score=0.0)
                for metric in self.metrics
            }

        return self._compute_metrics(
            query, expected_ids, retrieved_ids, expected_texts, retrieved_texts
        )

    def evaluate(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]] = None,
        mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
        **kwargs: Any,
    ) -> SegmentedRetrievalEvalResult:
        """
        Run evaluation with query string and expected ids.

        Args:
            query (str): Query string
            expected_ids (List[str]): Expected ids

        Returns:
            SegmentedRetrievalEvalResult: Evaluation result

        """
        result = super().evaluate(
            query=query,
            expected_ids=expected_ids,
            expected_texts=expected_texts,
            mode=mode,
            **kwargs,
        )
        return cast(SegmentedRetrievalEvalResult, result)

    async def aevaluate(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]] = None,
        mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
        **kwargs: Any,
    ) -> SegmentedRetrievalEvalResult:
        """
        Run evaluation, computing metrics overall and per segment.

        Retrieval runs exactly once; the segment breakdown reuses those nodes.
        """
        retrieved_nodes = await self.base_evaluator.aget_retrieved_nodes(query)
        retrieved_ids, retrieved_texts = _nodes_to_ids_and_texts(retrieved_nodes)

        metric_dict = self._compute_metrics(
            query, expected_ids, retrieved_ids, expected_texts, retrieved_texts
        )

        grouped_nodes = self._group_nodes_by_segment(retrieved_nodes)
        grouped_expected: Optional[Dict[str, Tuple[List[str], Optional[List[str]]]]] = (
            None
        )
        if self.expected_segments is not None:
            grouped_expected = self._group_expected_by_segment(
                self.expected_segments, expected_ids, expected_texts
            )
            # Relevant segments without retrieved nodes still need a score
            for segment in grouped_expected:
                grouped_nodes.setdefault(segment, [])

        segment_retrieved_ids: Dict[str, List[str]] = {}
        segment_metric_dict: Dict[str, Dict[str, RetrievalMetricResult]] = {}
        not_applicable_segments: List[str] = []
        for segment, nodes in grouped_nodes.items():
            segment_ids, segment_texts = _nodes_to_ids_and_texts(nodes)
            segment_retrieved_ids[segment] = segment_ids

            if grouped_expected is None:
                segment_expected_ids, segment_expected_texts = (
                    expected_ids,
                    expected_texts,
                )
            elif segment in grouped_expected:
                segment_expected_ids, segment_expected_texts = grouped_expected[segment]
            else:
                not_applicable_segments.append(segment)
                continue

            segment_metric_dict[segment] = self._compute_segment_metrics(
                query,
                segment_expected_ids,
                segment_ids,
                segment_expected_texts,
                segment_texts,
            )

        return SegmentedRetrievalEvalResult(
            query=query,
            expected_ids=expected_ids,
            expected_texts=expected_texts,
            retrieved_ids=retrieved_ids,
            retrieved_texts=retrieved_texts,
            mode=mode,
            metric_dict=metric_dict,
            segment_metric_dict=segment_metric_dict,
            segment_retrieved_ids=segment_retrieved_ids,
            not_applicable_segments=not_applicable_segments,
        )
