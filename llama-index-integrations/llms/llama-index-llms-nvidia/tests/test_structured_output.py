from pydantic import BaseModel
from typing import List, Any
from llama_index.llms.nvidia import NVIDIA as Interface
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.program import FunctionCallingProgram
import pytest
from llama_index.llms.nvidia.utils import (
    NVIDIA_FUNTION_CALLING_MODELS,
    API_CATALOG_MODELS,
)

from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    ChoiceLogprobs,
)
from openai.types.completion import Completion, CompletionUsage
from unittest.mock import MagicMock, patch


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]


prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""

llms = [
    "google/codegemma-1.1-7b",
    "google/codegemma-7b",
    "meta/codellama-70b",
    "meta/llama3-70b-instruct",
    "meta/llama3-8b-instruct",
]


def unsupported_models():
    return API_CATALOG_MODELS.keys() - NVIDIA_FUNTION_CALLING_MODELS


def mock_chat_completion_v1(model_name: str) -> ChatCompletion:
    response_content = """{
        "name": "Greatest Hits",
        "artist": "Best Artist",
        "songs": [
            {"title": "Hit Song 1", "length_seconds": 180},
            {"title": "Hit Song 2", "length_seconds": 210}
        ]
    }"""

    return ChatCompletion(
        id="chatcmpl-4162e407-e121-42b4-8590-1c173380be7d",
        object="chat.completion",
        created=1713474384,
        model=model_name,
        usage=CompletionUsage(
            completion_tokens=304, prompt_tokens=11, total_tokens=315
        ),
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=ChoiceLogprobs(
                    content=None,
                    text_offset=[],
                    token_logprobs=[0.0, 0.0],
                    tokens=[],
                    top_logprobs=[],
                ),
                message=ChatCompletionMessage(
                    content=response_content,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
    )


async def mock_async_chat_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return mock_chat_completion_v1(*args, **kwargs)


@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.parametrize("model", llms + list(NVIDIA_FUNTION_CALLING_MODELS))
def test_prompt_generation(MockSyncOpenAI: MagicMock, model):
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1(model)

    llm = Interface(model=model)
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Album, prompt_template_str=prompt_template_str, verbose=True, llm=llm
    )

    output = program(movie_name="The Shining")
    assert isinstance(output, Album), f"Expected Album, but got {type(output)}"
    assert isinstance(output.name, str), "Name should be a string"
    assert isinstance(output.artist, str), "artist should be a string"
    assert isinstance(output.songs, list), "Songs should be a list"
    assert all(
        isinstance(song, Song) for song in output.songs
    ), "All songs should be of type Song"

    assert len(output.songs) > 0, "Album should contain at least one song"


@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.parametrize("model", NVIDIA_FUNTION_CALLING_MODELS)
def test_empty_movie_name(MockSyncOpenAI: MagicMock, model):
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1(model)

    llm = Interface(model=model)
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Album, prompt_template_str=prompt_template_str, verbose=True, llm=llm
    )
    output = program(movie_name="")
    assert (
        len(output.songs) > 0
    ), "Album should contain at least one song even with an empty movie name"


@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.parametrize("model", unsupported_models())
def test_unsupported_models(MockSyncOpenAI: MagicMock, model: str):
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1(model)

    llm = Interface(model=model)

    with pytest.raises(ValueError) as e:
        FunctionCallingProgram.from_defaults(
            output_cls=Album,
            prompt_template_str=prompt_template_str,
            verbose=True,
            llm=llm,
        )
    assert f"{model} does not support function calling API." in str(e.value)


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.parametrize("model", llms + list(NVIDIA_FUNTION_CALLING_MODELS))
async def test_async_program(MockAsyncOpenAI: MagicMock, model) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion_v1(model)

    llm = Interface(model=model)
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Album, prompt_template_str=prompt_template_str, verbose=True, llm=llm
    )

    output = program(movie_name="The Shining")
    assert isinstance(output, Album), f"Expected Album, but got {type(output)}"
    assert isinstance(output.name, str), "Name should be a string"
    assert isinstance(output.artist, str), "artist should be a string"
    assert isinstance(output.songs, list), "Songs should be a list"
    assert all(
        isinstance(song, Song) for song in output.songs
    ), "All songs should be of type Song"

    assert len(output.songs) > 0, "Album should contain at least one song"
