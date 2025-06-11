"""CambAI text to speech tool spec."""

from typing import List, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class CambAIToolSpec(BaseToolSpec):
    """CambAI tool spec for text-to-speech synthesis."""

    spec_functions = ["get_voices", "text_to_speech"]

    def __init__(
        self, api_key: str, base_url: Optional[str] = "https://client.camb.ai/apis"
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): Your CambAI API key
            base_url (Optional[str]): The base url of CambAI

        """
        self.api_key = api_key
        self.base_url = base_url

    def get_voices(self) -> List[dict]:
        """
        Get list of available voices from CambAI.

        Returns:
            List[dict]: List of available voices with their details

        """
        from cambai import CambAI

        client = CambAI(api_key=self.api_key)
        voices = client.list_voices()
        return [voice.model_dump() for voice in voices]

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        voice_id: int = 20303,
    ) -> str:
        """
        Convert text to speech using CambAI API.

        Args:
            text (str): The text to convert to speech
            output_path (str): Path to save the audio file. If None, generates one
            voice_id (int): Override the default voice ID

        Returns:
            str: Path to the generated audio file

        """
        from cambai import CambAI
        from cambai.models.output_type import OutputType

        if output_path is None:
            output_path = f"cambai_speech.wav"
        client = CambAI(api_key=self.api_key)
        client.text_to_speech(
            text,
            voice_id=voice_id,
            output_type=OutputType.RAW_BYTES,
            save_to_file=output_path,
            verbose=True,
        )
        return output_path
