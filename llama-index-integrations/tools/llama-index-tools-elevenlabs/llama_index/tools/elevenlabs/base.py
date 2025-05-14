"""ElevenLabs text to speech tool spec."""

from typing import List, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ElevenLabsToolSpec(BaseToolSpec):
    """ElevenLabs tool spec for text-to-speech synthesis."""

    spec_functions = ["get_voices", "text_to_speech"]

    def __init__(
        self, api_key: str, base_url: Optional[str] = "https://api.elevenlabs.io"
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): Your ElevenLabs API key
            base_url (Optional[str]): The base url of elevenlabs

        """
        self.api_key = api_key
        self.base_url = base_url

    def get_voices(self) -> List[dict]:
        """
        Get list of available voices from ElevenLabs.

        Returns:
            List[dict]: List of available voices with their details

        """
        from elevenlabs import ElevenLabs

        # Create the client
        client = ElevenLabs(base_url=self.base_url, api_key=self.api_key)

        # Get the voices
        response = client.voices.get_all()

        # Return the dumped voice models as dict
        return [voice.model_dump() for voice in response.voices]

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
        voice_stability: Optional[float] = None,
        voice_similarity_boost: Optional[float] = None,
        voice_style: Optional[float] = None,
        voice_use_speaker_boost: Optional[bool] = None,
        model_id: Optional[str] = "eleven_monolingual_v1",
    ) -> str:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text (str): The text to convert to speech
            output_path (str): Where to save the output file
            output_path (str): Path to save the audio file. If None, generates one
            voice_id (Optional[str]): Override the default voice ID
            voice_stability (Optional[float]): The stability setting of the voice
            voice_similarity_boost (Optional[float]): The similarity boost setting of the voice
            voice_style: (Optional[float]): The style setting of the voice
            voice_use_speaker_boost (Optional[bool]): Whether to use speaker boost or not
            model_id (Optional[str]): Override the default model ID

        Returns:
            str: Path to the generated audio file

        """
        from elevenlabs import ElevenLabs, VoiceSettings
        from elevenlabs.client import DEFAULT_VOICE

        # Create client
        client = ElevenLabs(base_url=self.base_url, api_key=self.api_key)

        # Default the settings if not supplied
        if voice_stability is None:
            voice_stability = DEFAULT_VOICE.settings.stability

        if voice_similarity_boost is None:
            voice_similarity_boost = DEFAULT_VOICE.settings.similarity_boost

        if voice_style is None:
            voice_style = DEFAULT_VOICE.settings.style

        if voice_use_speaker_boost is None:
            voice_use_speaker_boost = DEFAULT_VOICE.settings.use_speaker_boost

        # Create the VoiceSettings
        voice_settings = VoiceSettings(
            stability=voice_stability,
            similarity_boost=voice_similarity_boost,
            style=voice_style,
            use_speaker_boost=voice_use_speaker_boost,
        )

        # Generate audio
        audio = client.generate(
            text=text, voice=voice_id, voice_settings=voice_settings, model=model_id
        )

        # Save the audio
        with open(output_path, "wb") as fp:
            fp.write(b"".join(audio))

        # Return the save location
        return output_path
