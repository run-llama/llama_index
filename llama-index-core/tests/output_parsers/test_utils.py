from llama_index.core.output_parsers.utils import _marshal_llm_to_json, extract_json_str


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


def test_marshal_llm_to_json_prefers_object_over_inline_citation() -> None:
    output = (
        "Based on document [1], here is the choice: "
        '{"choice": 2, "reason": "ok"}'
    )
    assert _marshal_llm_to_json(output) == '{"choice": 2, "reason": "ok"}'


def test_marshal_llm_to_json_ignores_non_json_bracket() -> None:
    output = 'Comparing [Model-A] vs [Model-B]: {"choice": 1}'
    assert _marshal_llm_to_json(output) == '{"choice": 1}'


def test_marshal_llm_to_json_keeps_array_of_objects() -> None:
    output = 'Output: [{"choice": 1}, {"choice": 2}]'
    assert _marshal_llm_to_json(output) == '[{"choice": 1}, {"choice": 2}]'