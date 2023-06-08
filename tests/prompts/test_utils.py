from llama_index.prompts.utils import convert_to_handlebars


def test_convert_to_handlebars() -> None:
    test_str = "This is a string with {variable} and {{key: value}}"
    expected_str = "This is a string with {{variable}} and {key: value}"

    assert convert_to_handlebars(test_str) == expected_str
