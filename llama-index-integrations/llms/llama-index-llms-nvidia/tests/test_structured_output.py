import respx
from httpx import Response
from pydantic import BaseModel
from typing import List
from llama_index.llms.nvidia import NVIDIA as Interface
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.program import FunctionCallingProgram
import pytest
from llama_index.llms.nvidia.utils import (
    NVIDIA_FUNTION_CALLING_MODELS,
    API_CATALOG_MODELS,
)


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


def create_mock_chat_completion_v1_response() -> dict:
    return {
        "id": "chatcmpl-4162e407-e121-42b4-8590-1c173380be7d",
        "object": "chat.completion",
        "created": 1713474384,
        "model": "mocked-model",
        "usage": {"completion_tokens": 304, "prompt_tokens": 11, "total_tokens": 315},
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {
                    "content": None,
                    "text_offset": [],
                    "token_logprobs": [0.0, 0.0],
                    "tokens": [],
                    "top_logprobs": [],
                },
                "message": {
                    "content": """{
                        "name": "Greatest Hits",
                        "artist": "Best Artist",
                        "songs": [
                            {"title": "Hit Song 1", "length_seconds": 180},
                            {"title": "Hit Song 2", "length_seconds": 210}
                        ]
                    }""",
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
    }


@respx.mock
@pytest.mark.parametrize("model", NVIDIA_FUNTION_CALLING_MODELS)
def test_prompt_generation(model):
    respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=Response(200, json=create_mock_chat_completion_v1_response())
    )

    llm = Interface(api_key="BOGUS", model=model)
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


@pytest.mark.parametrize(
    "model", API_CATALOG_MODELS.keys() - NVIDIA_FUNTION_CALLING_MODELS
)
def test_unsupported_models(model: str):
    llm = Interface(api_key="BOGUS", model=model)

    with pytest.raises(ValueError) as e:
        FunctionCallingProgram.from_defaults(
            output_cls=Album,
            prompt_template_str=prompt_template_str,
            verbose=True,
            llm=llm,
        )
    assert f"{model} does not support function calling API." in str(e.value)


@pytest.mark.asyncio()
@respx.mock
@pytest.mark.parametrize("model", NVIDIA_FUNTION_CALLING_MODELS)
async def test_async_program(model) -> None:
    respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=Response(200, json=create_mock_chat_completion_v1_response())
    )

    llm = Interface(api_key="BOGUS", model=model)
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
