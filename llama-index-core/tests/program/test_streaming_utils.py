"""Test streaming utilities."""

from typing import List, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.program.streaming_utils import (
    process_streaming_content_incremental,
    _extract_partial_list_progress,
    _parse_partial_list_items,
)


class Joke(BaseModel):
    """Test joke model."""

    setup: str
    punchline: Optional[str] = None


class Show(BaseModel):
    """Test show model with jokes list."""

    title: str = ""
    jokes: List[Joke] = Field(default_factory=list)


class Person(BaseModel):
    """Test person model."""

    name: str
    age: Optional[int] = None
    hobbies: List[str] = Field(default_factory=list)


def test_process_streaming_content_incremental_complete_json():
    """Test processing complete JSON content."""
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age": 30, "hobbies": ["reading", "coding"]}',
        )
    )

    result = process_streaming_content_incremental(response, Person)
    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 30
    assert result.hobbies == ["reading", "coding"]


def test_process_streaming_content_incremental_incomplete_json():
    """Test processing incomplete JSON content."""
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age": 30',
        )
    )

    result = process_streaming_content_incremental(response, Person)
    # Should handle incomplete JSON gracefully
    assert hasattr(result, "name")


def test_process_streaming_content_incremental_with_current_object():
    """Test processing with existing current object."""
    current_person = Person(name="Jane", age=25)

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age": 30}',
        )
    )

    result = process_streaming_content_incremental(response, Person, current_person)
    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 30


def test_process_streaming_content_incremental_empty_content():
    """Test processing empty content."""
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
        )
    )

    result = process_streaming_content_incremental(response, Person)
    # Should return FlexibleModel instance when no content
    assert hasattr(result, "__dict__")


def test_process_streaming_content_incremental_with_list():
    """Test processing content with list structures."""
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"title": "Comedy Show", "jokes": [{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}]}',
        )
    )

    result = process_streaming_content_incremental(response, Show)
    assert isinstance(result, Show)
    assert result.title == "Comedy Show"
    assert len(result.jokes) == 1
    assert result.jokes[0].setup == "Why did the chicken cross the road?"
    assert result.jokes[0].punchline == "To get to the other side!"


def test_process_streaming_content_incremental_malformed_json():
    """Test processing malformed JSON with current object."""
    current_show = Show(
        title="Comedy Show",
        jokes=[Joke(setup="First joke", punchline="First punchline")],
    )

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"jokes": [{"setup": "Second joke", "punchline": "Second punchline"}',
        )
    )

    result = process_streaming_content_incremental(response, Show, current_show)
    # Should attempt to extract partial progress
    assert hasattr(result, "jokes")


def test_extract_partial_list_progress_valid():
    """Test extracting partial list progress with valid content."""
    content = '{"jokes": [{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}]'
    current_show = Show(title="Comedy Show")

    from llama_index.core.program.utils import create_flexible_model

    partial_cls = create_flexible_model(Show)

    result = _extract_partial_list_progress(content, Show, current_show, partial_cls)

    if result is not None:
        assert hasattr(result, "jokes")


def test_extract_partial_list_progress_no_current():
    """Test extracting partial list progress without current object."""
    content = '{"jokes": [{"setup": "Why did the chicken cross the road?"}]'

    from llama_index.core.program.utils import create_flexible_model

    partial_cls = create_flexible_model(Show)

    result = _extract_partial_list_progress(content, Show, None, partial_cls)
    assert result is None


def test_extract_partial_list_progress_invalid_content():
    """Test extracting partial list progress with invalid content."""
    content = "invalid json content"
    current_show = Show(title="Comedy Show")

    from llama_index.core.program.utils import create_flexible_model

    partial_cls = create_flexible_model(Show)

    result = _extract_partial_list_progress(content, Show, current_show, partial_cls)
    assert result is None


def test_parse_partial_list_items_complete_objects():
    """Test parsing complete objects from list content."""
    list_content = '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}, {"setup": "Second joke", "punchline": "Second punchline"}'

    result = _parse_partial_list_items(list_content, "jokes", Show)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["setup"] == "Why did the chicken cross the road?"
    assert result[0]["punchline"] == "To get to the other side!"
    assert result[1]["setup"] == "Second joke"
    assert result[1]["punchline"] == "Second punchline"


def test_parse_partial_list_items_incomplete_objects():
    """Test parsing incomplete objects from list content."""
    list_content = '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}, {"setup": "Second joke"'

    result = _parse_partial_list_items(list_content, "jokes", Show)

    assert isinstance(result, list)
    # Should get at least the complete object
    assert len(result) >= 1
    assert result[0]["setup"] == "Why did the chicken cross the road?"


def test_parse_partial_list_items_invalid_content():
    """Test parsing invalid content returns empty list."""
    list_content = "completely invalid content"

    result = _parse_partial_list_items(list_content, "jokes", Show)

    assert isinstance(result, list)
    assert len(result) == 0


def test_parse_partial_list_items_empty_content():
    """Test parsing empty content returns empty list."""
    list_content = ""

    result = _parse_partial_list_items(list_content, "jokes", Show)

    assert isinstance(result, list)
    assert len(result) == 0


def test_parse_partial_list_items_malformed_json():
    """Test parsing malformed JSON objects."""
    list_content = '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}, {"setup": "Second joke", invalid'

    result = _parse_partial_list_items(list_content, "jokes", Show)

    assert isinstance(result, list)
    # Should get the complete object, ignore the malformed one
    assert len(result) >= 1
    assert result[0]["setup"] == "Why did the chicken cross the road?"


def test_process_streaming_content_incremental_none_message():
    """Test processing when message content is None."""
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
        )
    )

    result = process_streaming_content_incremental(response, Person)
    # Should return FlexibleModel instance when no content
    assert hasattr(result, "__dict__")


def test_process_streaming_content_incremental_progressive_list_building():
    """Test progressive list building with incremental updates."""
    # Start with empty show
    current_show = Show(title="Comedy Show", jokes=[])

    # First update - add one joke
    response1 = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"title": "Comedy Show", "jokes": [{"setup": "First joke", "punchline": "First punchline"}]}',
        )
    )

    result1 = process_streaming_content_incremental(response1, Show, current_show)
    assert len(result1.jokes) == 1
    assert result1.jokes[0].setup == "First joke"

    # Second update - add another joke
    response2 = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"title": "Comedy Show", "jokes": [{"setup": "First joke", "punchline": "First punchline"}, {"setup": "Second joke", "punchline": "Second punchline"}]}',
        )
    )

    result2 = process_streaming_content_incremental(response2, Show, result1)
    assert len(result2.jokes) == 2
    assert result2.jokes[1].setup == "Second joke"


def test_process_streaming_content_incremental_validation_error_fallback():
    """Test fallback when validation to target class fails."""
    # Create content that validates to FlexibleModel but not to strict Person
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "unknown_field": "value"}',
        )
    )

    result = process_streaming_content_incremental(response, Person)
    # Should return the flexible model instance when strict validation fails
    assert hasattr(result, "name")
    # Should still have the name field
    if hasattr(result, "name"):
        assert result.name == "John"
