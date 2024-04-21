from llama_index.core.evaluation.retrieval.metrics import HitRate, RR, MRR
import pytest


# Test cases using pytest
@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "expected_result"),
    [
        (["id1", "id2", "id3"], ["id3", "id1", "id2", "id4"], 1.0),
        (["id1", "id2", "id3", "id4"], ["id1", "id5", "id2"], 0.5),
        (["id1", "id2"], ["id3", "id4"], 0.0),
    ],
)
def test_hit_rate(expected_ids, retrieved_ids, expected_result):
    hr = HitRate()
    result = hr.compute(expected_ids, retrieved_ids)
    assert result.score == expected_result


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "expected_result"),
    [
        # Test cases that reflect the correct computation of RR
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            1 / 1,
        ),  # id3 is the first match, rank 1
        (
            ["id1", "id2", "id3", "id4"],
            ["id5", "id1"],
            1 / 2,
        ),  # id1 is the first match, rank 2
        (["id1", "id2"], ["id3", "id4"], 0.0),  # No matches found
        (
            ["id1", "id2"],
            ["id2", "id1", "id7"],
            1 / 1,
        ),  # id2 is the first match, rank 1
    ],
)
def test_rr(expected_ids, retrieved_ids, expected_result):
    rr = RR()
    result = rr.compute(expected_ids, retrieved_ids)
    assert result.score == pytest.approx(expected_result)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids", "expected_result"),
    [
        (
            ["id1", "id2", "id3"],
            ["id3", "id1", "id2", "id4"],
            (((1 / 1) + (1 / 2) + (1 / 3)) / 3),
        ),
        (
            ["id1", "id2", "id3", "id4"],
            ["id1", "id2", "id5"],
            (((1 / 1) + (1 / 2)) / 2),
        ),
        (["id1", "id2"], ["id3", "id4"], 0.0),
        (["id1", "id2"], ["id1", "id7", "id15", "id2"], (((1 / 1) + (1 / 4)) / 2)),
    ],
)
def test_mrr(expected_ids, retrieved_ids, expected_result):
    mrr = MRR()
    result = mrr.compute(expected_ids, retrieved_ids)
    assert result.score == pytest.approx(expected_result)


# NOTE: The following test cases are specifically for handling ValueErrors
@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids"),
    [
        ([], []),  # Empty IDs should trigger ValueError
        (
            None,
            ["id3", "id1", "id2", "id4"],
        ),  # None expected_ids should trigger ValueError
        (["id1", "id2", "id3"], None),  # None retrieved_ids should trigger ValueError
    ],
)
def test_hit_rate_exceptions(expected_ids, retrieved_ids):
    hr = HitRate()
    with pytest.raises(ValueError):
        hr.compute(expected_ids, retrieved_ids)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids"),
    [
        # Test cases for handling exceptions
        ([], []),  # Empty IDs should trigger ValueError
        (
            None,
            ["id3", "id1", "id2", "id4"],
        ),  # None expected_ids should trigger ValueError
        (["id1", "id2", "id3"], None),  # None retrieved_ids should trigger ValueError
    ],
)
def test_rr_exceptions(expected_ids, retrieved_ids):
    rr = RR()
    with pytest.raises(ValueError):
        rr.compute(expected_ids, retrieved_ids)


@pytest.mark.parametrize(
    ("expected_ids", "retrieved_ids"),
    [
        ([], []),  # Empty IDs should trigger ValueError
        (
            None,
            ["id3", "id1", "id2", "id4"],
        ),  # None expected_ids should trigger ValueError
        (["id1", "id2", "id3"], None),  # None retrieved_ids should trigger ValueError
    ],
)
def test_mrr_exceptions(expected_ids, retrieved_ids):
    mrr = MRR()
    with pytest.raises(ValueError):
        mrr.compute(expected_ids, retrieved_ids)
