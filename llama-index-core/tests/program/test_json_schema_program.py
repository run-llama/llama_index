"""Test JsonSchemaProgram."""

from unittest.mock import MagicMock, patch
import pytest
import json

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import BaseModel
from typing import List
from llama_index.core.program import JsonSchemaProgram


class MockSong(BaseModel):
    """Mock Song class."""

    title: str
    duration: int


class MockAlbum(BaseModel):
    """Mock Album class."""

    title: str
    artist: str
    songs: List[MockSong]


MOCK_ALBUM_JSON = {
    "title": "Test Album",
    "artist": "Test Artist",
    "songs": [
        {"title": "Song 1", "duration": 180},
        {"title": "Song 2", "duration": 210},
    ],
}


class MockLLM(MagicMock):
    """Mock LLM that returns JSON responses."""

    def chat(self, messages, **kwargs):
        """Mock chat method that returns JSON response."""
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=json.dumps(MOCK_ALBUM_JSON)
            )
        )

    async def achat(self, messages, **kwargs):
        """Mock async chat method that returns JSON response."""
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=json.dumps(MOCK_ALBUM_JSON)
            )
        )

    def _extend_messages(self, messages):
        """Mock extend messages."""
        return messages

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)


def test_json_schema_program_basic():
    """Test basic JsonSchemaProgram functionality."""
    prompt_template_str = "Generate an album about {topic}"

    with patch(
        "openai.resources.beta.chat.completions._type_to_response_format"
    ) as mock_format:
        # Mock the response format function
        mock_format.return_value = {
            "type": "json_schema",
            "json_schema": {"name": "MockAlbum"},
        }

        program = JsonSchemaProgram.from_defaults(
            output_cls=MockAlbum,
            prompt_template_str=prompt_template_str,
            llm=MockLLM(),
        )

        result = program(topic="rock music")

        # Verify the result
        assert isinstance(result, MockAlbum)
        assert result.title == "Test Album"
        assert result.artist == "Test Artist"
        assert len(result.songs) == 2
        assert result.songs[0].title == "Song 1"
        assert result.songs[0].duration == 180
        assert result.songs[1].title == "Song 2"
        assert result.songs[1].duration == 210


@pytest.mark.asyncio
async def test_json_schema_program_async():
    """Test async JsonSchemaProgram functionality."""
    prompt_template_str = "Generate an album about {topic}"

    with patch(
        "openai.resources.beta.chat.completions._type_to_response_format"
    ) as mock_format:
        # Mock the response format function
        mock_format.return_value = {
            "type": "json_schema",
            "json_schema": {"name": "MockAlbum"},
        }

        program = JsonSchemaProgram.from_defaults(
            output_cls=MockAlbum,
            prompt_template_str=prompt_template_str,
            llm=MockLLM(),
        )

        result = await program.acall(topic="jazz music")

        # Verify the result
        assert isinstance(result, MockAlbum)
        assert result.title == "Test Album"
        assert result.artist == "Test Artist"
        assert len(result.songs) == 2
        assert result.songs[0].title == "Song 1"
        assert result.songs[1].title == "Song 2"
