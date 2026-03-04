"""Test pydantic output parser."""

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock


class AttrDict(BaseModel):
    test_attr: str
    foo: int


class TestModel(BaseModel):
    __test__ = False
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
    assert "hello world" in formatted_query


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
