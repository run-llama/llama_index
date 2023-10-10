"""ElevenLabs TTS."""

from typing import Any, Optional

from llama_index.tts.base import BaseTTS


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs TTS.

    Args:
        api_key (Optional[str]): API key for ElevenLabs TTS.

    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()

        self.api_key = api_key

    def generate_audio(self, text: str, voice: Optional[str] = None) -> Any:
        """Generate audio.

        NOTE: return type is Any, but it should be any object that can be fed
        as `data` into IPython.display.Audio(). This includes numpy array, list,
        unicode, str or bytes

        Args:
            text (str): text to be turned into audio.
            voice (Optional[str]): voice in which audio is generated.
        """
        import_err_msg = "`elevenlabs` package not found, \
            please run `pip install elevenlabs`"

        try:
            import elevenlabs
        except ImportError:
            raise ImportError(import_err_msg)

        if self.api_key:
            elevenlabs.set_api_key(self.api_key)

        if voice:
            audio = elevenlabs.generate(text, voice=voice)
        else:
            audio = elevenlabs.generate(text)

        return audio
