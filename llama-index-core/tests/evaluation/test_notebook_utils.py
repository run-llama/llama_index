import pytest

from llama_index.core.evaluation.notebook_utils import get_retrieval_results_df
from llama_index.core.evaluation.retrieval.base import RetrievalEvalResult
from llama_index.core.evaluation.retrieval.metrics_base import RetrievalMetricResult


def make_result(score: float) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        query="query",
        expected_ids=["expected"],
        retrieved_ids=["retrieved"],
        retrieved_texts=["text"],
        metric_dict={
            "hit_rate": RetrievalMetricResult(score=score),
            "mrr": RetrievalMetricResult(score=score),
        },
    )


@pytest.mark.parametrize(("name_count", "result_count"), [(1, 2), (2, 1)])
def test_retrieval_results_reject_mismatched_lengths(
    name_count: int, result_count: int
) -> None:
    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="names and results_arr must have same length"):
        get_retrieval_results_df(
            [f"retriever-{i}" for i in range(name_count)],
            [[make_result(1.0)] for _ in range(result_count)],
        )


def test_retrieval_results_preserve_retriever_metric_alignment() -> None:
    pytest.importorskip("pandas")
    result = get_retrieval_results_df(
        ["first", "second"],
        [[make_result(0.0), make_result(1.0)], [make_result(1.0)]],
    )
    assert result.to_dict(orient="list") == {
        "retrievers": ["first", "second"],
        "hit_rate": [0.5, 1.0],
        "mrr": [0.5, 1.0],
    }


def test_retrieval_results_accept_empty_inputs() -> None:
    pytest.importorskip("pandas")
    result = get_retrieval_results_df([], [])
    assert result.empty
    assert list(result.columns) == ["retrievers"]
