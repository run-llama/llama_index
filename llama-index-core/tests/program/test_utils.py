"""Test program utils."""

import json
import pytest
from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.llms import MockFunctionCallingLLM
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


@pytest.mark.parametrize(
    ("partial_json", "expected"),
    [
        # unclosed arrays are closed in the order they were opened
        ('{"songs": [', {"songs": []}),
        ('{"tags": ["a", "b', {"tags": ["a", "b"]}),
        (
            '{"songs": [{"title": "Intro"',
            {"songs": [{"title": "Intro"}]},
        ),
        (
            '{"songs": [{"title": "Intro"}, {"title": "Ou',
            {"songs": [{"title": "Intro"}, {"title": "Ou"}]},
        ),
        # escaped quotes do not end the string
        ('{"artist": "The \\"Quoted', {"artist": 'The "Quoted'}),
        ('{"artist": "The \\"Quoted\\" Band', {"artist": 'The "Quoted" Band'}),
        # brackets inside strings are not structural
        ('{"title": "Intro {live', {"title": "Intro {live"}),
        ('{"title": "a]b', {"title": "a]b"}),
        ('{"title": "a}b"', {"title": "a}b"}),
        ('{"a": "]", "b": "[", "c": [', {"a": "]", "b": "[", "c": []}),
        # closed objects and arrays do not affect what is left open
        (
            '{"a": [[], {}], "b": [{"c": [1, [2]], "d": {}}, {"e": [',
            {"a": [[], {}], "b": [{"c": [1, [2]], "d": {}}, {"e": []}]},
        ),
        # non-ASCII text next to quotes and brackets
        ('{"name": "Über [live", "tags": ["é', {"name": "Über [live", "tags": ["é"]}),
        # a dangling escape character is dropped before closing the string
        ('{"path": "C:\\', {"path": "C:"}),
        # an escaped backslash is kept and does not escape the closing quote
        ('{"path": "C:\\\\', {"path": "C:\\"}),
        # a trailing comma between members/items is dropped
        ('{"name": "Echoes",', {"name": "Echoes"}),
        ('{"tags": ["a", ', {"tags": ["a"]}),
    ],
)
def test_repair_incomplete_json_produces_valid_json(
    partial_json: str, expected: Any
) -> None:
    """Repaired partial JSON should parse to the content streamed so far."""
    assert json.loads(_repair_incomplete_json(partial_json)) == expected


def test_process_streaming_objects_streams_list_items_incrementally() -> None:
    """Items of a list field should show up while the list is still streaming."""

    class Song(BaseModel):
        title: str
        length_seconds: int

    class Album(BaseModel):
        name: str
        songs: List[Song]

    full_json = json.dumps(
        {
            "name": "Echoes",
            "songs": [
                {"title": "Intro", "length_seconds": 61},
                {"title": "Outro", "length_seconds": 200},
            ],
        }
    )

    # Mirror LLMTextCompletionProgram.stream_call: feed every incomplete prefix
    # of the response and thread the previous result back in as `cur_objects`.
    cur_objects = None
    song_counts = []
    for end in range(1, len(full_json)):
        result = process_streaming_objects(
            CompletionResponse(text=full_json[:end]), Album, cur_objects=cur_objects
        )
        cur_objects = [result]
        song_counts.append(len(getattr(result, "songs", None) or []))

    # The first song must be visible before the second one finishes streaming,
    # instead of the whole list appearing only once the JSON is complete.
    assert 1 in song_counts
    assert song_counts.index(1) < song_counts.index(2)
    assert song_counts == sorted(song_counts)


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
    class MockLLM(MockFunctionCallingLLM):
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
