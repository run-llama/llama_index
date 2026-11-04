import pytest

from llama_index.core.llms.utils import parse_partial_json


@pytest.mark.parametrize(
    ("partial", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('{"a": "x', {"a": None}),
        ('{"a": "k: x\ny', {"a": "k: x\ny"}),
        ('{"a": "k: say \\"hi\\" {["}', {"a": 'k: say "hi" {['}),
        ('{"a": "k: back\\\\', {"a": "k: back\\"}),
        ('{"a": [1, {"b": "c"', {"a": [1, {"b": "c"}]}),
        ('{"a": 1, "b', {"a": 1}),
        ('{"a":', {"a": None}),
        ('{"a": 1,', {"a": 1}),
    ],
)
def test_parse_partial_json(partial: str, expected: dict) -> None:
    assert parse_partial_json(partial) == expected


@pytest.mark.parametrize(
    "partial", ['{"a": [1}', '{"a": "k: say \\"hi\\" {[', '{"a": "k: tail\\']
)
def test_parse_partial_json_malformed(partial: str) -> None:
    with pytest.raises(ValueError):
        parse_partial_json(partial)
