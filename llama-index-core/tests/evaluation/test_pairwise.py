import pytest

from llama_index.core.evaluation.pairwise import _default_parser_function


def test_final_verdict_wins_over_labels_mentioned_in_the_explanation():
    assert _default_parser_function(
        "I compared [[A]] and [[B]], and B is better. [[B]]"
    ) == (False, 0.0, "")
    assert _default_parser_function("Both [[A]] and [[B]] are equally good. [[C]]") == (
        None,
        0.5,
        "",
    )
    assert _default_parser_function("A is clearly better. [[A]]\n") == (True, 1.0, "")


@pytest.mark.parametrize(
    "response",
    [
        "[[B]]",
        "  [[B]]  \n",
        "The better answer is [[B]].",
        "**[[B]]**",
        "Verdict: [[B]] (assistant B gave more detail)",
    ],
)
def test_formatting_around_the_verdict_still_parses(response):
    """A judge that appends punctuation, Markdown or a parenthetical after the token still scores."""
    assert _default_parser_function(response) == (False, 0.0, "")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        # The old parser accepted any mention, so responses without a trailing token must keep parsing.
        ("[[A]] but then more explanation", (True, 1.0, "")),
        ("I would still choose [[A]] here.", (True, 1.0, "")),
        ("[[C]] either of them", (None, 0.5, "")),
    ],
)
def test_a_verdict_without_a_trailing_token_still_parses(response, expected):
    assert _default_parser_function(response) == expected


@pytest.mark.parametrize("response", ["", "no verdict here", "[[D]]", "[A]", "[[AB]]"])
def test_a_response_without_a_verdict_token_is_invalid(response):
    assert _default_parser_function(response) == (None, None, None)
