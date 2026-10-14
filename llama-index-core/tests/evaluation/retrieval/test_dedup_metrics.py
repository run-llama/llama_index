import pytest

from llama_index.core.evaluation.retrieval.metrics import AveragePrecision, NDCG


def test_ap_does_not_inflate_on_duplicate_retrieved_ids():
    metric = AveragePrecision()
    # The same relevant id appears twice in the retrieved list.
    result = metric.compute(expected_ids=["a"], retrieved_ids=["a", "a"])
    assert 0.0 <= result.score <= 1.0
    assert result.score == pytest.approx(1.0)


def test_ndcg_does_not_inflate_on_duplicate_retrieved_ids():
    metric = NDCG()
    result = metric.compute(expected_ids=["a"], retrieved_ids=["a", "a"])
    assert 0.0 <= result.score <= 1.0
    assert result.score == pytest.approx(1.0)


def test_unique_retrieved_ids_unchanged():
    ap = AveragePrecision().compute(
        expected_ids=["a", "b"], retrieved_ids=["a", "c", "b"]
    ).score
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)
