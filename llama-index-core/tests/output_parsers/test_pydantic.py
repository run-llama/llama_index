"""Test pydantic output parser."""

import json

import pytest
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.prompts import PromptTemplate


class AttrDict(BaseModel):
    test_attr: str
    foo: int


class TestModel(BaseModel):
    __test__ = False
    title: str
    attr_dict: AttrDict


class TestNonAsciiDescriptionModel(BaseModel):
    __test__ = False
    name: str = Field(description="用户名")  # Chinese for "username"
    formula: str = Field(..., examples=["H₂O"])
    currency: str = Field(..., examples=["€"])


def test_pydantic() -> None:
    """Test pydantic output parser."""
    output = """\

    Here is the valid JSON:
    {
        "title": "TestModel",
        "attr_dict": {
            "test_attr": "test_attr",
            "foo": 2
        }
    }
    """

    parser = PydanticOutputParser(output_cls=TestModel)
    parsed_output = parser.parse(output)
    assert isinstance(parsed_output, TestModel)
    assert parsed_output.title == "TestModel"
    assert isinstance(parsed_output.attr_dict, AttrDict)
    assert parsed_output.attr_dict.test_attr == "test_attr"
    assert parsed_output.attr_dict.foo == 2

    # TODO: figure out testing conditions
    with pytest.raises(ValueError):
        output = "hello world"
        parsed_output = parser.parse(output)


def test_pydantic_format() -> None:
    """Test pydantic format."""
    query = "hello world"
    parser = PydanticOutputParser(output_cls=AttrDict)
    formatted_query = parser.format(query)
    assert "hello world" in formatted_query


def test_pydantic_format_emits_valid_unescaped_schema() -> None:
    """format() must emit the real JSON schema, not a '{{'-escaped one."""
    parser = PydanticOutputParser(output_cls=AttrDict)
    formatted_query = parser.format("hello world")

    assert "{{" not in formatted_query

    # the embedded schema must be valid JSON
    schema_str = formatted_query.split("Here's a JSON schema to follow:")[1]
    schema_str = schema_str.split("Output a valid JSON object")[0].strip()
    schema = json.loads(schema_str)
    assert schema["title"] == "AttrDict"

    # the escaped variant is still available for embedding into raw templates
    assert "{{" in parser.format_string


def test_pydantic_output_parser_no_escaped_braces_in_final_prompt() -> None:
    """The final prompt built by PromptTemplate must contain a valid schema."""
    parser = PydanticOutputParser(output_cls=AttrDict)
    prompt = PromptTemplate("Extract the object: {text}", output_parser=parser)
    final_prompt = prompt.format(text="some text")

    assert "Extract the object: some text" in final_prompt
    assert "{{" not in final_prompt
    assert '"title": "AttrDict"' in final_prompt


def test_pydantic_format_with_blocks() -> None:
    """Test pydantic format with blocks."""
    parser = PydanticOutputParser(output_cls=AttrDict)
    messages = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="hello world"),
                ImageBlock(
                    url="https://pbs.twimg.com/media/GVhGD1PXkAANfPV?format=jpg&name=4096x4096"
                ),
                TextBlock(text="hello world"),
            ],
        )
    ]
    formatted_messages = parser.format_messages(messages)
    assert "hello world" in formatted_messages[0].blocks[-1].text


def test_pydantic_format_preserves_non_ascii_schema_descriptions() -> None:
    """Test pydantic format keeps non-ASCII schema descriptions readable."""
    query = "test"
    parser = PydanticOutputParser(output_cls=TestNonAsciiDescriptionModel)
    formatted_query = parser.format(query)

    assert "用户名" in formatted_query
    assert "\\u7528\\u6237\\u540d" not in formatted_query
    assert "H₂O" in formatted_query
    assert "H\\u2082O" not in formatted_query
    assert "€" in formatted_query
    assert "\\u20ac" not in formatted_query
