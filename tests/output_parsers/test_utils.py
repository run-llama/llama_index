from llama_index.output_parsers.utils import extract_json_str


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
