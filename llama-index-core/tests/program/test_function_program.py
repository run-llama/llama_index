"""Test LLM program."""

from unittest.mock import MagicMock
import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.bridge.pydantic import BaseModel
from typing import List, Optional, Union, Any
from llama_index.core.tools.types import BaseTool
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.tools import ToolOutput
from llama_index.core.program import FunctionCallingProgram


class MockSong(BaseModel):
    """Mock Song class."""

    title: str


class MockAlbum(BaseModel):
    title: str
    artist: str
    songs: List[MockSong]


MOCK_ALBUM = MockAlbum(
    title="hello",
    artist="world",
    songs=[MockSong(title="song1"), MockSong(title="song2")],
)

MOCK_ALBUM_2 = MockAlbum(
    title="hello2",
    artist="world2",
    songs=[MockSong(title="song3"), MockSong(title="song4")],
)


def _get_mock_album_response(
    allow_parallel_tool_calls: bool = False,
) -> AgentChatResponse:
    """Get mock album."""
    if allow_parallel_tool_calls:
        albums = [MOCK_ALBUM, MOCK_ALBUM_2]
    else:
        albums = [MOCK_ALBUM]

    tool_outputs = [
        ToolOutput(
            content=str(a),
            tool_name="tool_output",
            raw_input={},
            raw_output=a,
        )
        for a in albums
    ]

    # return tool outputs
    return AgentChatResponse(
        response="output",
        sources=tool_outputs,
    )


class MockLLM(MagicMock):
    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        return _get_mock_album_response(
            allow_parallel_tool_calls=allow_parallel_tool_calls
        )

    async def apredict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        return _get_mock_album_response(
            allow_parallel_tool_calls=allow_parallel_tool_calls
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)


def test_function_program() -> None:
    """Test Function program."""
    prompt_template_str = """This is a test album with {topic}"""
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str=prompt_template_str,
        llm=MockLLM(),
    )
    obj_output = llm_program(topic="songs")
    assert isinstance(obj_output, MockAlbum)
    assert obj_output.title == "hello"
    assert obj_output.artist == "world"
    assert obj_output.songs[0].title == "song1"
    assert obj_output.songs[1].title == "song2"


def test_function_program_multiple() -> None:
    """Test Function program multiple."""
    prompt_template_str = """This is a test album with {topic}"""
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str=prompt_template_str,
        llm=MockLLM(),
        allow_parallel_tool_calls=True,
    )
    obj_outputs = llm_program(topic="songs")
    assert isinstance(obj_outputs, list)
    assert len(obj_outputs) == 2
    assert isinstance(obj_outputs[0], MockAlbum)
    assert isinstance(obj_outputs[1], MockAlbum)
    # test second output
    assert obj_outputs[1].title == "hello2"
    assert obj_outputs[1].artist == "world2"
    assert obj_outputs[1].songs[0].title == "song3"
    assert obj_outputs[1].songs[1].title == "song4"


@pytest.mark.asyncio
async def test_async_function_program() -> None:
    """Test async function program."""
    # same as above but async
    prompt_template_str = """This is a test album with {topic}"""
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str=prompt_template_str,
        llm=MockLLM(),
    )
    obj_output = await llm_program.acall(topic="songs")
    assert isinstance(obj_output, MockAlbum)
    assert obj_output.title == "hello"
    assert obj_output.artist == "world"
    assert obj_output.songs[0].title == "song1"
    assert obj_output.songs[1].title == "song2"
