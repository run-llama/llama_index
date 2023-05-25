import pytest

from llama_index.llm_predictor.vellum.utils import convert_to_kebab_case


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("HelloWorld", "helloworld"),
        (
            "LlamaIndex Demo: query_keyword_extract",
            "llamaindex-demo-query-keyword-extract",
        ),
    ],
)
def test_convert_to_kebab_case(input_string: str, expected: str) -> None:
    assert convert_to_kebab_case(input_string) == expected
