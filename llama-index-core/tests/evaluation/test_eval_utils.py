import math

import pytest

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.eval_utils import get_results_df


def test_get_results_df_ignores_missing_and_nan_scores():
    systems = {
        "all_scored": [5.0, 5.0, 5.0, 5.0],
        "one_none": [5.0, 5.0, 5.0, None],
        "one_nan": [5.0, 5.0, 5.0, float("nan")],
    }
    df = get_results_df(
        [
            {"correctness": [EvaluationResult(score=score) for score in scores]}
            for scores in systems.values()
        ],
        list(systems),
        ["correctness"],
    )

    means = dict(zip(df["names"], df["correctness"]))
    assert means["all_scored"] == pytest.approx(5.0)
    assert means["one_none"] == pytest.approx(5.0)
    assert means["one_nan"] == pytest.approx(5.0)
    assert df["correctness_n_missing"].tolist() == [0, 1, 1]


def test_get_results_df_reports_all_missing_scores_as_nan():
    df = get_results_df(
        [
            {
                "correctness": [
                    EvaluationResult(score=None),
                    EvaluationResult(score=float("nan")),
                ]
            }
        ],
        ["unparsed"],
        ["correctness"],
    )

    assert math.isnan(df["correctness"].iloc[0])
    assert df["correctness_n_missing"].iloc[0] == 2
