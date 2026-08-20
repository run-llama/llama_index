from llama_index.core.prompts.utils import SafeFormatter, get_template_vars


def test_get_template_vars() -> None:
    template = "hello {text} {foo}"
    template_vars = get_template_vars(template)
    assert template_vars == ["text", "foo"]


def test_safe_formatter_unescapes_braces() -> None:
    """Escaped braces ("{{"/"}}") should collapse to literal braces, like str.format."""
    formatter = SafeFormatter(format_dict={"x": "X"})

    # "{{" -> "{" and "}}" -> "}", even when the escaped content is not a key
    # (matching the behavior of Python's str.format).
    template = "{{'head': '', 'props': {{...}}}} and {x}"
    assert formatter.format(template) == template.format(x="X")
    assert formatter.format(template) == "{'head': '', 'props': {...}} and X"

    # Empty escaped braces.
    assert formatter.format("{{}}") == "{}"

    # Escaped braces must not be treated as a field even when they wrap a valid key.
    assert formatter.format("{{x}}") == "{x}"

    # Missing keys are still left untouched (safe formatting, no KeyError).
    assert formatter.format("hi {missing}") == "hi {missing}"
