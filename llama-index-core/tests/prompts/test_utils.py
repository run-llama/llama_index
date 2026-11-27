from llama_index.core.prompts.utils import format_string, get_template_vars


def test_get_template_vars() -> None:
    template = "hello {text} {foo}"
    template_vars = get_template_vars(template)
    assert template_vars == ["text", "foo"]


def test_format_string_unescapes_doubled_braces() -> None:
    """Doubled braces are escapes and render as single braces, like str.format."""
    template = 'Answer {question} as JSON, for example {{"answer": "yes"}}.'
    assert (
        format_string(template, question="Q")
        == 'Answer Q as JSON, for example {"answer": "yes"}.'
    )


def test_format_string_missing_var_inside_escapes_stays_literal() -> None:
    """{{name}} is a literal {name}, not a field, even when name is not given."""
    assert format_string("{{not_a_var}}") == "{not_a_var}"


def test_format_string_escapes_adjacent_to_var() -> None:
    assert format_string("{{{var}}}", var="V") == "{V}"
    assert format_string("{{{{x}}}}") == "{{x}}"


def test_format_string_value_braces_are_not_rescanned() -> None:
    assert format_string("{x}", x='{"a": 1}') == '{"a": 1}'
    assert format_string("{x}", x="{{y}}") == "{{y}}"


def test_format_string_keeps_unescaped_nested_json() -> None:
    """A bare '}}' can be the end of nested JSON, not an escape."""
    template = 'Return JSON like {"id": 1, "meta": {}} for: {q}'
    assert (
        format_string(template, q="Q")
        == 'Return JSON like {"id": 1, "meta": {}} for: Q'
    )

    three_levels = 'Return JSON like {"a": {"b": {"c": 1}}} for: {q}'
    assert (
        format_string(three_levels, q="Q")
        == 'Return JSON like {"a": {"b": {"c": 1}}} for: Q'
    )

    assert format_string('[{"a": {}}, {"b": {}}]') == '[{"a": {}}, {"b": {}}]'


def test_format_string_round_trips_escaped_json() -> None:
    """Escaping every brace (as selection.py does) still round trips exactly."""
    import json

    value = {"answer": "yes", "meta": {}, "nested": {"a": {"b": [1, {"c": 2}]}}}
    escaped = json.dumps(value).replace("{", "{{").replace("}", "}}")

    rendered = format_string(escaped)

    assert json.loads(rendered) == value
