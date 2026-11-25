"""Tests for eval_utils.get_results_df score aggregation."""

import warnings

import numpy as np

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.eval_utils import get_results_df


def _df_for(systems: dict, metric: str = "correctness"):
    return get_results_df(
        [
            {metric: [EvaluationResult(score=s) for s in scores]}
            for scores in systems.values()
        ],
        list(systems),
        [metric],
    )


def test_mean_uses_scored_results_only() -> None:
    """A None score is a missing score, not a real 0.0."""
    df = _df_for(
        {
            "all_scored": [5.0, 5.0, 5.0, 5.0],
            "one_missing": [5.0, 5.0, 5.0, None],
            "one_zero": [5.0, 5.0, 5.0, 0.0],
        }
    )
    by_name = df.set_index("names")["correctness"]
    assert by_name["all_scored"] == 5.0
    # Missing score excluded: mean of the three real scores, not 3.75.
    assert by_name["one_missing"] == 5.0
    # A genuine 0.0 score still counts.
    assert by_name["one_zero"] == 3.75


def test_nan_score_does_not_poison_mean() -> None:
    df = _df_for({"one_nan": [5.0, 5.0, 5.0, float("nan")]})
    assert df["correctness"][0] == 5.0


def test_missing_scores_emit_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _df_for({"one_missing": [5.0, 5.0, 5.0, None]})
    assert any("missing" in str(w.message).lower() for w in caught)


def test_all_missing_gives_nan_mean() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _df_for({"none_scored": [None, None]})
    assert np.isnan(df["correctness"][0])


def test_no_warning_when_all_scored() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _df_for({"all_scored": [5.0, 5.0, 5.0, 5.0]})
    assert not caught
