"""Test pydantic output parser."""

from llama_index.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel
import pytest


class AttrDict(BaseModel):
    test_attr: str
    foo: int


class TestModel(BaseModel):
    title: str
    attr_dict: AttrDict


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
    expected_dict = (
        '{\n  "title": "AttrDict",\n  "test_attr": "str",\n  "foo": "int"\n}'
    )
    assert formatted_query == f"hello world\n\n{expected_dict}"
