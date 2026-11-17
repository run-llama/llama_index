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


def test_missing_or_nonterminal_verdict_is_invalid():
    assert _default_parser_function("[[A]] but then more explanation") == (
        None,
        None,
        None,
    )
    assert _default_parser_function("no verdict here") == (None, None, None)
