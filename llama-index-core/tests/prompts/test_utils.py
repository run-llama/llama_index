import ast

import pytest

from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_DYNAMIC_EXTRACT_PROMPT
from llama_index.core.prompts.utils import SafeFormatter, get_template_vars


def test_get_template_vars() -> None:
    template = "hello {text} {foo}"
    template_vars = get_template_vars(template)
    assert template_vars == ["text", "foo"]


@pytest.mark.parametrize(
    ("template", "kwargs", "expected"),
    [
        # Plain braces around non-field text used to be left doubled, so the
        # JSON examples in default prompts reached the model malformed.
        (
            "{{'head': '', 'props': {{...}}}} and {x}",
            {"x": "X"},
            "{'head': '', 'props': {...}} and X",
        ),
        ("{{}}", {}, "{}"),
        ("{{{{literal}}}}", {}, "{{literal}}"),
        # Escaped braces must not be substituted even when they wrap a real key.
        ("{{x}}", {"x": "V"}, "{x}"),
        ("{{{x}}}", {"x": "V"}, "{V}"),
        # Still safe: unknown keys are left alone instead of raising KeyError.
        ("hi {missing}", {}, "hi {missing}"),
        ("hi {missing} {x}", {"x": "X"}, "hi {missing} X"),
    ],
)
def test_safe_formatter_escaped_braces(
    template: str, kwargs: dict, expected: str
) -> None:
    assert SafeFormatter(format_dict=kwargs).format(template) == expected


def test_prompt_template_matches_str_format() -> None:
    template = 'Answer {question} as JSON, for example {{"answer": "yes"}}.'
    assert PromptTemplate(template).format(question="Q") == template.format(
        question="Q"
    )


def test_default_prompts_carry_unescaped_json() -> None:
    # DynamicLLMPathExtractor writes its output format with escaped braces; what
    # the model sees has to be a plain Python literal, not "{{...}}".
    prompt = DEFAULT_DYNAMIC_EXTRACT_PROMPT.format(
        text="T",
        max_knowledge_triplets=3,
        allowed_entity_types="[]",
        allowed_relation_types="[]",
    )
    line = next(
        line
        for line in prompt.splitlines()
        if line.startswith("- Output in JSON format")
    )
    payload = line.split("JSON format: ", 1)[1]
    assert "{{" not in payload and "}}" not in payload
    assert ast.literal_eval(payload) == [
        {"head": "", "head_type": "", "relation": "", "tail": "", "tail_type": ""}
    ]
