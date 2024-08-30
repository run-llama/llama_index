from pydantic import BaseModel
from typing import List
from llama_index.llms.nvidia import NVIDIA
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

llms = [
    "google/codegemma-1.1-7b",
    "google/codegemma-7b",
    "meta/codellama-70b",
    "meta/llama3-70b-instruct",
    "meta/llama3-8b-instruct",
]


def unsupported_models():
    return API_CATALOG_MODELS.keys() - NVIDIA_FUNTION_CALLING_MODELS


@pytest.mark.parametrize("key", llms)
def test_prompt_generation(key):
    llm = NVIDIA(model=key)
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

    assert output.name != "", "Album title should not be empty"
    assert len(output.songs) > 0, "Album should contain at least one song"


@pytest.mark.parametrize("key", NVIDIA_FUNTION_CALLING_MODELS)
def test_empty_movie_name(key):
    llm = NVIDIA(model=key)
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Album, prompt_template_str=prompt_template_str, verbose=True, llm=llm
    )
    output = program(movie_name="")
    assert (
        len(output.songs) > 0
    ), "Album should contain at least one song even with an empty movie name"


@pytest.mark.parametrize("key", unsupported_models())
def test_warning_unsupported_models(key: str):
    llm = NVIDIA(model=key)

    with pytest.raises(ValueError) as e:
        FunctionCallingProgram.from_defaults(
            output_cls=Album,
            prompt_template_str=prompt_template_str,
            verbose=True,
            llm=llm,
        )
    assert f"{key} does not support function calling API." in str(e.value)
