import pytest
from llama_index.core.evaluation.retrieval.metrics import HitRate, MRR


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

        mrr = MRR()
        mrr.use_granular_mrr = use_granular
        mrr.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
