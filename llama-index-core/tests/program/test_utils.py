"""Test program utils."""

import pytest
from typing import List, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.program.utils import (
    _repair_incomplete_json,
    process_streaming_objects,
    num_valid_fields,
    create_flexible_model,
)


class Person(BaseModel):
    name: str
    age: Optional[int] = None
    hobbies: List[str] = Field(default_factory=list)


def test_repair_incomplete_json() -> None:
    """Test JSON repair function."""
    # Test adding missing quotes
    assert _repair_incomplete_json('{"name": "John') == '{"name": "John"}'

    # Test adding missing braces
    assert _repair_incomplete_json('{"name": "John"') == '{"name": "John"}'

    # Test empty string
    assert _repair_incomplete_json("") == "{}"

    # Test already valid JSON
    valid_json = '{"name": "John", "age": 30}'
    assert _repair_incomplete_json(valid_json) == valid_json


def test_process_streaming_objects() -> None:
    """Test processing streaming objects."""
    # Test processing complete object
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age": 30}',
        )
    )

    result = process_streaming_objects(response, Person)
    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 30

    # Test processing incomplete object
    incomplete_response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age":',
        )
    )

    # Should return empty object when can't parse
    result = process_streaming_objects(incomplete_response, Person)
    assert result.name is None  # Default value

    # Test with previous state
    prev_obj = Person(name="John", age=25)
    result = process_streaming_objects(
        incomplete_response, Person, cur_objects=[prev_obj]
    )
    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 25  # Keeps previous state

    # Test with tool calls
    tool_call_response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "function": {
                            "name": "create_person",
                            "arguments": '{"name": "Jane", "age": 28}',
                        }
                    }
                ]
            },
        )
    )

    # Mock LLM for tool calls
    class MockLLM:
        def get_tool_calls_from_response(self, *args, **kwargs):
            return [
                type(
                    "ToolSelection",
                    (),
                    {"tool_kwargs": {"name": "Jane", "age": 28}},
                )
            ]

    result = process_streaming_objects(
        tool_call_response,
        Person,
        llm=MockLLM(),  # type: ignore
    )
    assert isinstance(result, Person)
    assert result.name == "Jane"
    assert result.age == 28


def test_num_valid_fields() -> None:
    """Test counting valid fields."""
    # Test simple object
    person = Person(name="John", age=None, hobbies=[])
    assert num_valid_fields(person) == 1  # Only name is non-None

    # Test with more fields
    person = Person(name="John", age=30, hobbies=["reading"])
    assert num_valid_fields(person) == 3  # All fields are non-None

    # Test list of objects
    people = [
        Person(name="John", age=30),
        Person(name="Jane", hobbies=["reading"]),
    ]
    assert num_valid_fields(people) == 4  # 2 names + 1 age + 1 hobby list

    # Test nested object
    class Family(BaseModel):
        parent: Person
        children: List[Person] = []

    family = Family(
        parent=Person(name="John", age=40),
        children=[Person(name="Jane", age=10)],
    )
    assert num_valid_fields(family) == 4  # parent's name & age + child's name & age


def test_create_flexible_model() -> None:
    """Test creating flexible model."""
    FlexiblePerson = create_flexible_model(Person)

    # Should accept partial data
    flexible_person = FlexiblePerson(name="John")
    assert flexible_person.name == "John"
    assert flexible_person.age is None

    # Should accept extra fields
    flexible_person = FlexiblePerson(
        name="John", extra_field="value", another_field=123
    )
    assert flexible_person.name == "John"
    assert hasattr(flexible_person, "extra_field")
    assert flexible_person.extra_field == "value"

    # Original model should still be strict
    with pytest.raises(ValueError):
        Person(name=None)  # type: ignore
