import json

import pytest
from llama_index.core.llms.utils import parse_partial_json


@pytest.mark.parametrize(
    ("partial", "expected"),
    [
        # already-valid JSON is returned untouched
        ("{}", {}),
        ('{"a": 1}', {"a": 1}),
        ('{"a": [1, 2]}', {"a": [1, 2]}),
        # an unterminated string value keeps the characters received so far
        ('{"a": "abc', {"a": "abc"}),
        ('{"a": "', {"a": ""}),
        ('{"a": ""', {"a": ""}),
        ('{"outer": {"inner": "wor', {"outer": {"inner": "wor"}}),
        ('{"a": {"b": [{"c": "de', {"a": {"b": [{"c": "de"}]}}),
        # ... including the last element of an array
        ('{"items": ["alpha", "bet', {"items": ["alpha", "bet"]}),
        ('["par', ["par"]),
        # ... and strings whose content looks structural
        ('{"a": "has: colon', {"a": "has: colon"}),
        ('{"a": "has {brace', {"a": "has {brace"}),
        ('{"a": "has \\"quote', {"a": 'has "quote'}),
        ('{"a": "back\\\\', {"a": "back\\"}),
        ('{"a": "line1\nline2', {"a": "line1\nline2"}),
        # an escape sequence cut mid-way is dropped, the rest is kept
        ('{"a": "back\\', {"a": "back"}),
        ('{"a": "uni \\u12', {"a": "uni "}),
        # a key with no value yet carries no information, so it is dropped
        ('{"a": 1, "b', {"a": 1}),
        ('{"a": 1, "b"', {"a": 1}),
        ('{"ke', {}),
        ('{"a"', {}),
        ("{", {}),
        ("[", []),
        # a value that has not started yet is null
        ('{"a":', {"a": None}),
        # ... as is a literal cut mid-token, which cannot be guessed
        ('{"a": 1.', {"a": None}),
        ('{"a": -', {"a": None}),
        ('{"a": 1e', {"a": None}),
        ('{"a": tru', {"a": None}),
        # a trailing comma is dropped
        ('{"a": 1,', {"a": 1}),
        ("[1,", [1]),
        # a complete literal is kept
        ('{"a": 12', {"a": 12}),
        ('{"a": [[1, 2], [3', {"a": [[1, 2], [3]]}),
    ],
)
def test_parse_partial_json(partial: str, expected: object) -> None:
    assert parse_partial_json(partial) == expected


@pytest.mark.parametrize(
    "malformed",
    ["", "   ", "}", "]", "{]", '{"a": 1}}', "[1, 2]]", '{"a" "b"}', "gibberish"],
)
def test_parse_partial_json_raises_on_malformed(malformed: str) -> None:
    with pytest.raises(ValueError):
        parse_partial_json(malformed)


STREAMED_TOOL_ARGS = [
    '{"query": "who won the 2026 world cup?", "top_k": 5, "rerank": true}',
    '{"path": "C:\\\\Users\\\\me\\\\notes.txt", "mode": "r"}',
    '{"messages": [{"role": "user", "content": "hi \\"there\\""}], "stream": false}',
    '{"filters": {"and": [{"key": "year", "op": ">=", "value": 2024}]}, "limit": null}',
    '{"text": "line one\\nline two", "emoji": "\\u2764\\ufe0f done"}',
    '[{"a": 1}, {"b": [true, false, null]}]',
]


def _assert_consistent_with(partial: object, final: object) -> None:
    """Every value in a partial parse must be a prefix of the final value."""
    if partial is None:
        # a value that has not started streaming yet
        return
    if isinstance(final, dict):
        assert isinstance(partial, dict)
        for key, value in partial.items():
            assert key in final, f"invented key {key!r}"
            _assert_consistent_with(value, final[key])
    elif isinstance(final, list):
        assert isinstance(partial, list)
        assert len(partial) <= len(final), "invented list element"
        for got, want in zip(partial, final):
            _assert_consistent_with(got, want)
    elif isinstance(final, str):
        assert isinstance(partial, str) and final.startswith(partial), (
            f"{partial!r} is not a prefix of {final!r}"
        )


@pytest.mark.parametrize("doc", STREAMED_TOOL_ARGS)
def test_parse_partial_json_over_every_prefix(doc: str) -> None:
    """Simulate a tool call streamed one character at a time."""
    final = json.loads(doc)
    for end in range(1, len(doc) + 1):
        _assert_consistent_with(parse_partial_json(doc[:end]), final)
    assert parse_partial_json(doc) == final
