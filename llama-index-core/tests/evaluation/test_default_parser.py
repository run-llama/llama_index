from llama_index.core.evaluation.eval_utils import default_parser


def test_default_parser_single_line_score():
    assert default_parser("4.0") == (4.0, "")


def test_default_parser_single_line_feedback_without_score():
    assert default_parser("Score: 4") == (None, "Score: 4")


def test_default_parser_multi_line_response():
    assert default_parser("4.0\nGood answer") == (4.0, "Good answer")
    assert default_parser("Score: 4\nFeedback") == (None, "Feedback")
    assert default_parser("4.0\n\nFeedback") == (4.0, "Feedback")


def test_default_parser_empty_response():
    assert default_parser("   ") == (None, "No response")
