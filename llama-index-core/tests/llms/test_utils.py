from llama_index.core.llms.utils import parse_partial_json


def test_parse_partial_json_keeps_incomplete_string_value() -> None:
    parsed = parse_partial_json('{"query": "hello')

    assert parsed == {"query": "hello"}


def test_parse_partial_json_drops_incomplete_key() -> None:
    parsed = parse_partial_json('{"query": "ok", "incomp')

    assert parsed == {"query": "ok"}
