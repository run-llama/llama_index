"""Test LLM program."""

from unittest.mock import MagicMock
import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import List, Optional, Union, Any, Dict
from llama_index.core.tools.types import BaseTool
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.tools import ToolOutput
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools.calling import call_tool


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
    last_predict_kwargs: Optional[Dict[str, Any]] = None
    last_apredict_kwargs: Optional[Dict[str, Any]] = None

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
        self.last_predict_kwargs = kwargs
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
        self.last_apredict_kwargs = kwargs
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


def test_function_program_forwards_tool_choice() -> None:
    """Test Function program passes tool_choice into predict_and_call."""
    llm = MockLLM()
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=llm,
        tool_choice={"type": "any"},
    )
    llm_program(topic="songs")
    assert llm.last_predict_kwargs is not None
    assert llm.last_predict_kwargs["tool_choice"] == {"type": "any"}


@pytest.mark.asyncio
async def test_async_function_program_forwards_tool_choice() -> None:
    """Test async Function program passes tool_choice into apredict_and_call."""
    llm = MockLLM()
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=llm,
        tool_choice={"type": "any"},
    )
    await llm_program.acall(topic="songs")
    assert llm.last_apredict_kwargs is not None
    assert llm.last_apredict_kwargs["tool_choice"] == {"type": "any"}


# Tests for issue #21024: single-field models with defaults


class SingleFieldListModel(BaseModel):
    """Single field list model with default."""

    names: List[str] = Field(default_factory=list, description="List of names")


class SingleFieldStrModel(BaseModel):
    """Single field string model with default."""

    value: str = Field(default="", description="A string value")


class SingleFieldIntModel(BaseModel):
    """Single field int model with default."""

    count: int = Field(default=0, description="A count value")


def test_single_field_list_with_default() -> None:
    """
    Test that single-field list model with default receives correct value.

    Regression test for https://github.com/run-llama/llama_index/issues/21024
    """
    tool = get_function_tool(SingleFieldListModel)
    arguments = {"names": ["deep learning", "neural networks"]}
    result = call_tool(tool, arguments)

    assert result.raw_output.names == ["deep learning", "neural networks"]


def test_single_field_str_with_default() -> None:
    """Test that single-field string model with default receives correct value."""
    tool = get_function_tool(SingleFieldStrModel)
    arguments = {"value": "hello world"}
    result = call_tool(tool, arguments)

    assert result.raw_output.value == "hello world"


def test_single_field_int_with_default() -> None:
    """Test that single-field int model with default receives correct value."""
    tool = get_function_tool(SingleFieldIntModel)
    arguments = {"count": 42}
    result = call_tool(tool, arguments)

    assert result.raw_output.count == 42


# Tests for Structured outputs sometimes return string instead of Pydantic model
# Used opus here for tests


class MockLLMReturnsNoToolCalls(MagicMock):
    """Mock LLM that returns a text response with no tool calls (empty sources)."""

    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        return AgentChatResponse(
            response="I can help with that! Here's a great album.",
            sources=[],
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
        return AgentChatResponse(
            response="I can help with that! Here's a great album.",
            sources=[],
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)


class MockLLMReturnsErrorToolOutput(MagicMock):
    """Mock LLM that returns a ToolOutput with is_error=True (tool call failed)."""

    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        error = ValueError("1 validation error for MockAlbum\nfield required")
        tool_output = ToolOutput(
            content="Encountered error: 1 validation error for MockAlbum",
            tool_name="MockAlbum",
            raw_input={"title": "hello"},
            raw_output=str(error),
            is_error=True,
            exception=error,
        )
        return AgentChatResponse(
            response="Encountered error",
            sources=[tool_output],
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
        error = ValueError("1 validation error for MockAlbum\nfield required")
        tool_output = ToolOutput(
            content="Encountered error: 1 validation error for MockAlbum",
            tool_name="MockAlbum",
            raw_input={"title": "hello"},
            raw_output=str(error),
            is_error=True,
            exception=error,
        )
        return AgentChatResponse(
            response="Encountered error",
            sources=[tool_output],
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)


class MockLLMReturnsStringRawOutput(MagicMock):
    """Mock LLM that returns a ToolOutput where raw_output is a string (not a model)."""

    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        tool_output = ToolOutput(
            content="some text output",
            tool_name="MockAlbum",
            raw_input={},
            raw_output="this is a string, not a pydantic model",
        )
        return AgentChatResponse(
            response="some text output",
            sources=[tool_output],
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
        tool_output = ToolOutput(
            content="some text output",
            tool_name="MockAlbum",
            raw_input={},
            raw_output="this is a string, not a pydantic model",
        )
        return AgentChatResponse(
            response="some text output",
            sources=[tool_output],
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)


def test_function_program_raises_on_no_tool_calls() -> None:
    """
    Test that FunctionCallingProgram raises a clear ValueError when the LLM
    returns no tool calls (empty sources).

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsNoToolCalls(),
    )
    with pytest.raises(ValueError, match="did not return any tool calls"):
        llm_program(topic="songs")


@pytest.mark.asyncio
async def test_async_function_program_raises_on_no_tool_calls() -> None:
    """
    Test async: FunctionCallingProgram raises a clear ValueError when the LLM
    returns no tool calls.

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsNoToolCalls(),
    )
    with pytest.raises(ValueError, match="did not return any tool calls"):
        await llm_program.acall(topic="songs")


def test_function_program_raises_on_tool_error() -> None:
    """
    Test that FunctionCallingProgram raises a clear ValueError when a tool call
    fails (is_error=True on ToolOutput), instead of silently returning a string.

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsErrorToolOutput(),
    )
    with pytest.raises(ValueError, match="Structured output extraction failed"):
        llm_program(topic="songs")


@pytest.mark.asyncio
async def test_async_function_program_raises_on_tool_error() -> None:
    """
    Test async: FunctionCallingProgram raises a clear ValueError when a tool
    call fails.

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsErrorToolOutput(),
    )
    with pytest.raises(ValueError, match="Structured output extraction failed"):
        await llm_program.acall(topic="songs")


def test_function_program_raises_on_string_raw_output() -> None:
    """
    Test that FunctionCallingProgram raises a clear ValueError when raw_output
    is a string instead of a Pydantic model.

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsStringRawOutput(),
    )
    with pytest.raises(ValueError, match="expected a MockAlbum instance but got str"):
        llm_program(topic="songs")


@pytest.mark.asyncio
async def test_async_function_program_raises_on_string_raw_output() -> None:
    """
    Test async: FunctionCallingProgram raises a clear ValueError when raw_output
    is a string instead of a Pydantic model.

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsStringRawOutput(),
    )
    with pytest.raises(ValueError, match="expected a MockAlbum instance but got str"):
        await llm_program.acall(topic="songs")


def test_function_program_error_message_includes_llm_text() -> None:
    """
    Test that the error message for no-tool-call includes the LLM's text response
    so the user can see what the model actually returned.

    """
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=MockAlbum,
        prompt_template_str="This is a test album with {topic}",
        llm=MockLLMReturnsNoToolCalls(),
    )
    with pytest.raises(ValueError, match="Here's a great album"):
        llm_program(topic="songs")
