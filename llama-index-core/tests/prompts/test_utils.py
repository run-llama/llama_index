from llama_index.core.prompts.utils import get_template_vars


def test_get_template_vars() -> None:
    template = "hello {text} {foo}"
    template_vars = get_template_vars(template)
    assert template_vars == ["text", "foo"]


def test_safe_formatter_unescapes_doubled_braces() -> None:
    """{{ and }} must collapse to literal braces like str.format does."""
    from llama_index.core.prompts.utils import SafeFormatter

    formatter = SafeFormatter({"name": "bob"})
    assert formatter.format("Say {{hi}} and {name}") == "Say {hi} and bob"
    assert formatter.format("json: {{}}") == "json: {}"
    json_formatter = SafeFormatter({"query": "text", "top_k": 3})
    assert (
        json_formatter.format('example: {{"query": "{query}", "top_k": {top_k}}}')
        == 'example: {"query": "text", "top_k": 3}'
    )


def test_safe_formatter_keeps_unknown_and_malformed_fields() -> None:
    from llama_index.core.prompts.utils import SafeFormatter

    formatter = SafeFormatter({"known": "v"})
    assert formatter.format("{unknown}") == "{unknown}"
    assert formatter.format("no closing brace {here") == "no closing brace {here"
    assert formatter.format("nested {outer {inner}} brace") == (
        "nested {outer {inner}} brace"
    )
    assert formatter.format("value with braces: {known}") == "value with braces: v"


def test_safe_formatter_does_not_reprocess_substituted_values() -> None:
    from llama_index.core.prompts.utils import SafeFormatter

    formatter = SafeFormatter({"payload": "{{not_an_escape}}"})
    assert formatter.format("data: {payload}") == "data: {{not_an_escape}}"
