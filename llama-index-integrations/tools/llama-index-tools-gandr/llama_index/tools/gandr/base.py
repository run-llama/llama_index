"""Gandr text to speech tool spec."""

import os
from typing import List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://tts.gandr.ai"
DEFAULT_MODEL = "tts-1"
DEFAULT_VOICE = "gandr-mia"
DEFAULT_RESPONSE_FORMAT = "mp3"
MAX_INPUT_CHARACTERS = 2000
GANDR_VOICES = (
    "gandr-mia",
    "gandr-ava",
    "gandr-jenny",
    "gandr-dane",
    "gandr-leo",
    "gandr-lewis",
)


class GandrToolSpec(BaseToolSpec):
    """Gandr tool spec for text-to-speech synthesis."""

    spec_functions = ["get_voices", "text_to_speech"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = DEFAULT_BASE_URL,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (Optional[str]): Your Gandr API key. Falls back to the
                GANDR_API_KEY environment variable when not provided.
            base_url (Optional[str]): The base url of the Gandr API

        """
        self.api_key = api_key or os.environ.get("GANDR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Gandr API key found. Pass api_key or set the "
                "GANDR_API_KEY environment variable."
            )
        self.base_url = base_url

    def get_voices(self) -> List[str]:
        """
        Get the list of available Gandr voices.

        Returns:
            List[str]: Names of the available voices

        """
        return list(GANDR_VOICES)

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = DEFAULT_VOICE,
        response_format: Optional[str] = DEFAULT_RESPONSE_FORMAT,
        model: Optional[str] = DEFAULT_MODEL,
    ) -> str:
        """
        Convert text to speech using the Gandr API.

        Args:
            text (str): The text to convert to speech. At most 2000
                characters per request.
            output_path (str): Where to save the output file
            voice (Optional[str]): The voice to use, one of get_voices()
            response_format (Optional[str]): One of "mp3", "wav" or "pcm".
                "pcm" is headerless s16le mono at 24000 Hz.
            model (Optional[str]): The model to use

        Returns:
            str: Path to the generated audio file

        """
        import requests

        if len(text) > MAX_INPUT_CHARACTERS:
            raise ValueError(
                f"Input is {len(text)} characters; the Gandr API accepts "
                f"at most {MAX_INPUT_CHARACTERS} characters per request. "
                "Split the text into smaller requests."
            )

        # The response body streams, so write it to the file in chunks
        response = requests.post(
            f"{self.base_url}/v1/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": response_format,
            },
            stream=True,
        )
        response.raise_for_status()

        # Save the audio
        with open(output_path, "wb") as fp:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    fp.write(chunk)

        # Return the save location
        return output_path
