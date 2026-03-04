from pydantic import BaseModel
from typing import Any, AsyncGenerator, List
from llama_index.llms.nvidia import NVIDIA as Interface
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.program import FunctionCallingProgram
import pytest
from llama_index.llms.nvidia.utils import (
    MODEL_TABLE,
)
from openai.types.completion import Completion, CompletionUsage
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    ChoiceLogprobs,
)
from unittest.mock import MagicMock, patch

NVIDIA_STRUCT_OUT_MODELS = []
for model in MODEL_TABLE.values():
    if model.supports_structured_output:
        NVIDIA_STRUCT_OUT_MODELS.append(model.id)


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


def create_mock_chat_completion_v1_response(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-4162e407-e121-42b4-8590-1c173380be7d",
        object="chat.completion",
        model=model,
        created=1713474384,
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
                    content="""{
                            "name": "Greatest Hits",
                            "artist": "Best Artist",
                            "songs": [
                                {"title": "Hit Song 1", "length_seconds": 180},
                                {"title": "Hit Song 2", "length_seconds": 210}
                            ]
                        }""",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
    )


async def mock_async_chat_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[Completion, None]:
    async def gen() -> AsyncGenerator[Completion, None]:
        for response in create_mock_chat_completion_v1_response(*args, **kwargs):
            yield response

    return gen()


# @respx.mock
@patch("llama_index.llms.openai.base.SyncOpenAI")
@pytest.mark.parametrize("model", NVIDIA_STRUCT_OUT_MODELS)
def test_prompt_generation(MockSyncOpenAI: MagicMock, model):
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = (
        create_mock_chat_completion_v1_response(model)
    )

    llm = Interface(api_key="BOGUS", model=model)
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Album, prompt_template_str=prompt_template_str, verbose=True, llm=llm
    )
    assert llm.metadata is not None

    output = program(movie_name="The Shining")
    assert isinstance(output, Album), f"Expected Album, but got {type(output)}"
    assert isinstance(output.name, str), "Name should be a string"
    assert isinstance(output.artist, str), "artist should be a string"
    assert isinstance(output.songs, list), "Songs should be a list"
    assert all(isinstance(song, Song) for song in output.songs), (
        "All songs should be of type Song"
    )

    assert len(output.songs) > 0, "Album should contain at least one song"


@pytest.mark.parametrize("model", MODEL_TABLE.keys() - NVIDIA_STRUCT_OUT_MODELS)
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
