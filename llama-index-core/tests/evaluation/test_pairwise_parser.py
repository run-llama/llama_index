import pytest

from llama_index.core.evaluation.pairwise import _default_parser_function


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        # Test cases for earlier tokens must not override final verdict
        ("Considered [[A]], chose [[B]]", (False, 0.0, "")),
        ("Compared [[A]] and [[B]], tie [[C]]", (None, 0.5, "")),
        ("Initially [[B]], finally [[A]]", (True, 1.0, "")),
        # Test cases for tolerance, formatting that endswith() would reject
        ("Verdict: `[[B]]`", (False, 0.0, "")),
        ("[[B]] followed by explanation", (False, 0.0, "")),
        ("[[C]]\nThank you.", (None, 0.5, "")),
        ("Explanation [[A]] \n", (True, 1.0, "")),
        ("[[A]] ... [[B]] ... [[C]]", (None, 0.5, "")),
        # Test cases for regression, valid and invalid input behavior
        ("[[A]]", (True, 1.0, "")),
        ("[[B]]", (False, 0.0, "")),
        ("[[C]]", (None, 0.5, "")),
        ("No verdict", (None, None, None)),
        ("", (None, None, None)),
        ("[A]", (None, None, None)),
        ("[[D]]", (None, None, None)),
        ("Answer A is better", (None, None, None)),
    ],
)
def test_pairwise_parser(response: str, expected: tuple) -> None:
    assert _default_parser_function(response) == expected
