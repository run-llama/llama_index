import pytest
from llama_index.core.evaluation.eval_utils import default_parser


@pytest.mark.parametrize(
    ("eval_response", "expected_score", "expected_reasoning"),
    [
        # score + reasoning on separate lines (the documented format)
        ("4.5\nThe answer is accurate.", 4.5, "The answer is accurate."),
        # score only, no reasoning line -- used to raise ValueError
        ("4.5", 4.5, ""),
        ("  4.5  ", 4.5, ""),
        # leading blank line -- score used to be read as "" and lost
        ("\n4.5\nThe answer is accurate.", 4.5, "The answer is accurate."),
        # empty response
        ("", None, "No response"),
        ("   ", None, "No response"),
        # unparseable score
        ("not a number\nsome reasoning", None, "some reasoning"),
    ],
)
def test_default_parser(eval_response, expected_score, expected_reasoning):
    score, reasoning = default_parser(eval_response)
    assert score == expected_score
    assert reasoning == expected_reasoning
