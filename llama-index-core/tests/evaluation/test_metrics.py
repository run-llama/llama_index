from math import log2

import pytest
from llama_index.core.evaluation.retrieval.metrics import (
    AveragePrecision,
    HitRate,
    MRR,
    NDCG,
    Precision,
    Recall,
)


# Test cases for the updated HitRate class using instance attribute
@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "use_granular", "expected_result"),
    [
        (["id1", "id2", "id3"], ["id3", "id1", "id2", "id4"], False, 1.0),
        (["id1", "id2", "id3", "id4"], ["id1", "id5", "id2"], True, 2 / 4),
        (["id1", "id2"], ["id3", "id4"], False, 0.0),
        (["id1", "id2"], ["id2", "id1", "id7"], True, 2 / 2),
    ],
)
def test_hit_rate(expected_ids, retrieved_ids, use_granular, expected_result):
    hr = HitRate()
    hr.use_granular_hit_rate = use_granular
    result = hr.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    assert result.score == pytest.approx(expected_result)


# Test cases for the updated MRR class using instance attribute
@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "use_granular", "expected_result"),
    [
        (["id1", "id2", "id3"], ["id3", "id1", "id2", "id4"], False, 1 / 1),
        (["id1", "id2", "id3", "id4"], ["id5", "id1"], False, 1 / 2),
        (["id1", "id2"], ["id3", "id4"], False, 0.0),
        (["id1", "id2"], ["id2", "id1", "id7"], False, 1 / 1),
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            True,
            (1 / 1 + 1 / 2 + 1 / 3) / 3,
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id1", "id2", "id5"],
            True,
            (1 / 1 + 1 / 2) / 2,
        ),
        (["id1", "id2"], ["id1", "id7", "id15", "id2"], True, (1 / 1 + 1 / 4) / 2),
    ],
)
def test_mrr(expected_ids, retrieved_ids, use_granular, expected_result):
    mrr = MRR()
    mrr.use_granular_mrr = use_granular
    result = mrr.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    assert result.score == pytest.approx(expected_result)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "expected_result"),
    [
        (["id1", "id2", "id3"], ["id3", "id1", "id2", "id4"], 3 / 4),
        (["id1", "id2", "id3", "id4"], ["id5", "id1"], 1 / 2),
        (["id1", "id2"], ["id3", "id4"], 0 / 2),
        (["id1", "id2"], ["id2", "id1", "id7"], 2 / 3),
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            3 / 4,
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id1", "id2", "id5"],
            2 / 3,
        ),
        (["id1", "id2"], ["id1", "id7", "id15", "id2"], 2 / 4),
    ],
)
def test_precision(expected_ids, retrieved_ids, expected_result):
    prec = Precision()
    result = prec.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    assert result.score == pytest.approx(expected_result)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "expected_result"),
    [
        (["id1", "id2", "id3"], ["id3", "id1", "id2", "id4"], 3 / 3),
        (["id1", "id2", "id3", "id4"], ["id5", "id1"], 1 / 4),
        (["id1", "id2"], ["id3", "id4"], 0 / 2),
        (["id1", "id2"], ["id2", "id1", "id7"], 2 / 2),
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            3 / 3,
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id1", "id2", "id5"],
            2 / 4,
        ),
        (["id1", "id2"], ["id1", "id7", "id15", "id2"], 2 / 2),
    ],
)
def test_recall(expected_ids, retrieved_ids, expected_result):
    recall = Recall()
    result = recall.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    assert result.score == pytest.approx(expected_result)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "expected_result"),
    [
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            (1 / 1 + 2 / 2 + 3 / 3 + 0) / 3,
        ),
        (["id1", "id2", "id3", "id4"], ["id5", "id1"], (0 + 1 / 2) / 4),
        (["id1", "id2"], ["id3", "id4"], (0 + 0) / 2),
        (["id1", "id2"], ["id2", "id1", "id7"], (1 / 1 + 2 / 2 + 0) / 2),
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            (1 / 1 + 2 / 2 + 3 / 3 + 0) / 3,
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id1", "id2", "id5"],
            (1 / 1 + 2 / 2 + 0) / 4,
        ),
        (
            ["id1", "id2"],
            ["id1", "id7", "id15", "id2"],
            (1 / 1 + 0 + 0 + 2 / 4) / 2,
        ),
    ],
)
def test_ap(expected_ids, retrieved_ids, expected_result):
    ap = AveragePrecision()
    result = ap.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    assert result.score == pytest.approx(expected_result)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "mode", "expected_result"),
    [
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            "linear",
            (1 / log2(1 + 1) + 1 / log2(2 + 1) + 1 / log2(3 + 1))
            / (1 / log2(1 + 1) + 1 / log2(2 + 1) + 1 / log2(3 + 1)),
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id5", "id1"],
            "linear",
            (1 / log2(2 + 1))
            / (1 / log2(1 + 1) + 1 / log2(2 + 1) + 1 / log2(3 + 1) + 1 / log2(4 + 1)),
        ),
        (
            ["id1", "id2"],
            ["id3", "id4"],
            "linear",
            0.0,
        ),
        (
            ["id1", "id2"],
            ["id2", "id1", "id7"],
            "linear",
            (1 / log2(1 + 1) + 1 / log2(2 + 1)) / (1 / log2(1 + 1) + 1 / log2(2 + 1)),
        ),
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            "exponential",
            (1 / log2(1 + 1) + 1 / log2(2 + 1) + 1 / log2(3 + 1))
            / (1 / log2(1 + 1) + 1 / log2(2 + 1) + 1 / log2(3 + 1)),
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id1", "id2", "id5"],
            "exponential",
            (1 / log2(1 + 1) + 1 / log2(2 + 1))
            / (1 / log2(1 + 1) + 1 / log2(2 + 1) + 1 / log2(3 + 1) + 1 / log2(4 + 1)),
        ),
        (
            ["id1", "id2"],
            ["id1", "id7", "id15", "id2"],
            "exponential",
            (1 / log2(1 + 1) + 1 / log2(4 + 1)) / (1 / log2(1 + 1) + 1 / log2(2 + 1)),
        ),
    ],
)
def test_ndcg(expected_ids, retrieved_ids, mode, expected_result):
    ndcg = NDCG()
    ndcg.mode = mode
    result = ndcg.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    assert result.score == pytest.approx(expected_result)


# Test cases for exceptions handling for both HitRate and MRR
@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "use_granular"),
    [
        (
            None,
            ["id3", "id1", "id2", "id4"],
            False,
        ),  # None expected_ids should trigger ValueError
        (
            ["id1", "id2", "id3"],
            None,
            True,
        ),  # None retrieved_ids should trigger ValueError
        ([], [], False),  # Empty IDs should trigger ValueError
    ],
)
def test_exceptions(expected_ids, retrieved_ids, use_granular):
    with pytest.raises(ValueError):
        hr = HitRate()
        hr.use_granular_hit_rate = use_granular
        hr.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

    with pytest.raises(ValueError):
        mrr = MRR()
        mrr.use_granular_mrr = use_granular
        mrr.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

    with pytest.raises(ValueError):
        prec = Precision()
        prec.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

    with pytest.raises(ValueError):
        recall = Recall()
        recall.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

    with pytest.raises(ValueError):
        ap = AveragePrecision()
        ap.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

    with pytest.raises(ValueError):
        ndcg = NDCG()
        ndcg.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
