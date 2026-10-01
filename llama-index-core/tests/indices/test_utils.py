"""Test indices/utils.py."""

import pytest
from llama_index.core.indices.utils import expand_tokens_with_subtokens


def test_expand_tokens_with_subtokens() -> None:
    """Test expand tokens."""
    tokens = {"foo bar", "baz", "hello hello world bye"}
    keywords = expand_tokens_with_subtokens(tokens)
    assert keywords == {
        "foo bar",
        "foo",
        "bar",
        "baz",
        "hello hello world bye",
        "hello",
        "world",
        "bye",
    }


parse_choice_test_lines = [
    """ Doc: 2, Relevance: 8 (The document mentions taking a "tasty turn around Barcelona\'s Santa Caterina market" and listening to an episode about Barcelona.)\nDoc: 4, Relevance: 6 (The document mentions Ferramenta in Barcelona and recommends cocktails and pasta dishes that can be tried there.)""",
    "Doc: 2, Relevance: 8\nDoc: 4, Relevance: 6",
    "answer_num: 2, answer_relevance:8\nanswer_num: 4, answer_relevance:6",
]


@pytest.mark.parametrize("answer", parse_choice_test_lines)
def test_default_parse_choice_select_answer_fn(answer):
    from llama_index.core.indices.utils import default_parse_choice_select_answer_fn

    answer_nums, answer_relevances = default_parse_choice_select_answer_fn(answer, 5)
    assert answer_nums == [2, 4]
    assert answer_relevances == [8, 6]


@pytest.mark.parametrize(
    ("answer", "expected_nums", "expected_relevances"),
    [
        # A line whose relevance cannot be parsed must be skipped as a whole,
        # otherwise its choice number is kept without a matching relevance and
        # the two returned lists no longer line up.
        (
            "answer_num: 1, answer_relevance: 7\n"
            "answer_num: 2, answer_relevance: high\n"
            "answer_num: 3, answer_relevance: 3",
            [1, 3],
            [7, 3],
        ),
        (
            "answer_num: 1, answer_relevance: 7\nanswer_num: 2, answer_relevance: 5",
            [1, 2],
            [7, 5],
        ),
    ],
)
def test_default_parse_choice_select_answer_fn_keeps_lists_aligned(
    answer, expected_nums, expected_relevances
):
    from llama_index.core.indices.utils import default_parse_choice_select_answer_fn

    answer_nums, answer_relevances = default_parse_choice_select_answer_fn(answer, 5)

    assert answer_nums == expected_nums
    assert answer_relevances == expected_relevances
    assert len(answer_nums) == len(answer_relevances)
