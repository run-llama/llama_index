"""Tests for the default PairwiseComparisonEvaluator response parser."""

import pytest

from llama_index.core.evaluation.pairwise import _default_parser_function


@pytest.mark.parametrize(
    ("eval_response", "expected"),
    [
        ("[[A]]", (True, 1.0, "")),
        ("[[B]]", (False, 0.0, "")),
        ("[[C]]", (None, 0.5, "")),
    ],
)
def test_bare_verdict(eval_response: str, expected: tuple) -> None:
    assert _default_parser_function(eval_response) == expected


@pytest.mark.parametrize(
    ("eval_response", "expected"),
    [
        # The explanation names the other assistant before the final verdict.
        ("I compared [[A]] and [[B]], and B is better. [[B]]", (False, 0.0, "")),
        ("Both [[A]] and [[B]] are equally good. [[C]]", (None, 0.5, "")),
        (
            "[[B]] was weaker on accuracy, so the better answer is [[A]]",
            (True, 1.0, ""),
        ),
        # A tie mentioned mid-explanation must not win over the final verdict.
        ("This is not a [[C]] tie: A is clearly better. [[A]]", (True, 1.0, "")),
    ],
)
def test_final_verdict_wins_over_labels_in_the_explanation(
    eval_response: str, expected: tuple
) -> None:
    """The prompt asks for the explanation first and the verdict after it."""
    assert _default_parser_function(eval_response) == expected


@pytest.mark.parametrize(
    "eval_response",
    [
        "",
        "The responses are similar.",
        "Assistant A is better.",
        "[[D]]",
        "[A]",
    ],
)
def test_a_response_with_no_verdict_is_unparsed(eval_response: str) -> None:
    assert _default_parser_function(eval_response) == (None, None, None)


@pytest.mark.parametrize(
    "eval_response",
    [
        "The better answer is [[B]].",
        "The better answer is [[B]]\n\n",
        "**[[B]]**",
        "Verdict: [[B]] (assistant B gave more detail)",
    ],
)
def test_a_verdict_is_still_read_with_text_after_it(eval_response: str) -> None:
    """
    Why the last label, and not a strict check that the response ends with one.

    Judges routinely add a closing period, markdown emphasis or a parenthetical
    after the label. Requiring the response to end with the verdict would score
    all of these as unparsed, which `_get_eval_result` reports as a failed
    evaluation. All four of these are read correctly today, so that stricter
    check would be a regression.
    """
    assert _default_parser_function(eval_response) == (False, 0.0, "")
