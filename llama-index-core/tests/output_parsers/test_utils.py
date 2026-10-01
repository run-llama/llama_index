from llama_index.core.output_parsers.utils import (
    _marshal_llm_to_json,
    extract_json_str,
    parse_json_markdown,
)


def test_extract_json_str() -> None:
    input = """\
Here is the valid JSON:
{
    "title": "TestModel",
    "attr_dict": {
        "test_attr": "test_attr",
        "foo": 2
    }
}\
"""
    expected = """\
{
    "title": "TestModel",
    "attr_dict": {
        "test_attr": "test_attr",
        "foo": 2
    }
}\
"""
    assert extract_json_str(input) == expected


def test_marshal_object_after_prose_with_square_brackets() -> None:
    # An LLM commonly emits prose containing '[' before the real JSON object;
    # the leading bracket must not hijack extraction.
    text = 'See [1] for details: {"answer": 42}'
    assert _marshal_llm_to_json(text) == '{"answer": 42}'
    assert parse_json_markdown(text) == {"answer": 42}


def test_marshal_object_with_inner_array_after_prose() -> None:
    text = 'Sure! [thinking] Here: {"a": [1, 2], "b": 3}'
    assert _marshal_llm_to_json(text) == '{"a": [1, 2], "b": 3}'
    assert parse_json_markdown(text) == {"a": [1, 2], "b": 3}
