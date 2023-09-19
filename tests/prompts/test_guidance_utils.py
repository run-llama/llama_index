from typing import List

from llama_index.bridge.pydantic import BaseModel

from llama_index.prompts.guidance_utils import (
    convert_to_handlebars,
    pydantic_to_guidance_output_template,
)


def test_convert_to_handlebars() -> None:
    test_str = "This is a string with {variable} and {{key: value}}"
    expected_str = "This is a string with {{variable}} and {key: value}"

    assert convert_to_handlebars(test_str) == expected_str


class TestSimpleModel(BaseModel):
    attr0: str
    attr1: str


EXPECTED_SIMPLE_STR = """\
{
  "attr0": "{{gen 'attr0' stop='"'}}",
  "attr1": "{{gen 'attr1' stop='"'}}",
}\
"""


class TestNestedModel(BaseModel):
    attr2: List[TestSimpleModel]


EXPECTED_NESTED_STR = """\
{
  "attr2": [{{#geneach 'attr2' stop=']'}}{{#unless @first}}, {{/unless}}{
  "attr0": "{{gen 'attr0' stop='"'}}",
  "attr1": "{{gen 'attr1' stop='"'}}",
}{{/geneach}}],
}\
"""


def test_convert_pydantic_to_guidance_output_template_simple() -> None:
    output_str = pydantic_to_guidance_output_template(TestSimpleModel)
    assert output_str == EXPECTED_SIMPLE_STR


def test_convert_pydantic_to_guidance_output_template_nested() -> None:
    output_str = pydantic_to_guidance_output_template(TestNestedModel)
    assert output_str == EXPECTED_NESTED_STR
