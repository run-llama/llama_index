from llama_index.core.output_parsers.utils import extract_json_str, parse_code_markdown


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


def test_python_language_marker_not_leaked_when_fence_not_at_start() -> None:
    text = "Here is the code:\n```python\nx = 1\n```"
    result = parse_code_markdown(text, only_last=False)
    assert len(result) == 1
    assert "python" not in result[0]
    assert result[0].strip() == "x = 1"


def test_python_language_marker_not_leaked_multiple_blocks() -> None:
    text = "```python\na = 1\n```\nand then\n```python\nb = 2\n```"
    result = parse_code_markdown(text, only_last=False)
    assert [r.strip() for r in result] == ["a = 1", "b = 2"]
    assert all("python" not in r for r in result)
