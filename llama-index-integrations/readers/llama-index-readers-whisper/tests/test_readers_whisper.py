import requests
import pytest
import os
from llama_index.core.readers.base import BaseReader
from llama_index.readers.whisper import WhisperReader
from io import BytesIO


AUDIO_URL = "https://science.nasa.gov/wp-content/uploads/2024/04/sounds-of-mars-one-small-step-earth.wav"
AUDIO_URL = "https://audio-samples.github.io/samples/mp3/blizzard_primed/sample-0.mp3"
OPENAI_AVAILABLE = os.getenv("OPENAI_API_KEY") is not None


def test_class():
    names_of_base_classes = [b.__name__ for b in WhisperReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_get_file_or_bytes():
    reader = WhisperReader(model="whisper-1", api_key="test")
    audio_bytes = requests.get(AUDIO_URL).content
    file_path_or_bytes = reader._get_file_path_or_bytes(audio_bytes)
    assert isinstance(file_path_or_bytes, BytesIO)


def test_get_file_or_bytes_file():
    reader = WhisperReader(model="whisper-1", api_key="test")
    audio_bytes = requests.get(AUDIO_URL).content
    # Create a temporary file-like object with a name
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    file_path_or_bytes = reader._get_file_path_or_bytes(audio_file)
    assert isinstance(file_path_or_bytes, BytesIO)


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI API key not available")
def test_load_data_bytes():
    reader = WhisperReader(model="whisper-1")
    audio_bytes = requests.get(AUDIO_URL).content

    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    documents = reader.load_data(audio_file)
    assert len(documents) == 1
    assert documents[0].text is not None
    assert documents[0].metadata is not None


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI API key not available")
def test_load_data_file():
    reader = WhisperReader(model="whisper-1")
    audio_file = requests.get(AUDIO_URL)
    with open("test_audio.mp3", "wb") as f:
        f.write(audio_file.content)

    documents = reader.load_data("test_audio.mp3")
    assert len(documents) == 1
    assert documents[0].text is not None
    assert documents[0].metadata is not None


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI API key not available")
@pytest.mark.asyncio
async def test_load_data_async_bytes():
    reader = WhisperReader(model="whisper-1")
    audio_bytes = requests.get(AUDIO_URL).content
    documents = await reader.aload_data(audio_bytes)
    assert len(documents) == 1
    assert documents[0].text is not None
    assert documents[0].metadata is not None
