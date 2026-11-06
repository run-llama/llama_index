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
