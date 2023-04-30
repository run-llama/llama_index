"""Text to speech module."""

from typing import Optional
import tempfile
import os
import numpy as np

# text to be chunked into chunks of 10 words
# to avoid hallicunation for bark
DEFAULT_CHUNK_SIZE = 10


class BaseTTS:
    def __init__(self) -> None:
        pass

    def generate_audio(self, text: str) -> None:
        raise NotImplementedError(
            "generate_audio method should be implemented by subclasses"
        )


class BarkTTS(BaseTTS):
    def __init__(
        self,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        lang_speaker_voice: Optional[str] = None,
    ) -> None:
        """
        Args:
            text_temp: generation temperature (1.0 more diverse, \
                0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, \
                0.0 more conservative)
            lang_speaker_voice: language speaker voice for audio cloning.
        """

        super().__init__()

        self.text_temp = text_temp
        self.waveform_temp = waveform_temp
        self.lang_speaker_voice = lang_speaker_voice

    def generate_audio(self, text: str) -> None:
        """
        Args:
            text: text to be turned into audio.
        """

        import_err_msg = "`bark` package not found, \
            please run `pip install git+https://github.com/suno-ai/bark.git`"
        try:
            import bark
        except ImportError:
            raise ImportError(import_err_msg)

        words = text.split()
        chunks = [
            words[i : i + DEFAULT_CHUNK_SIZE]
            for i in range(0, len(words), DEFAULT_CHUNK_SIZE)
        ]
        chunks = [" ".join(chunk) for chunk in chunks]  # type: ignore

        full_generation = None
        history_prompt = self.lang_speaker_voice
        audio_chunks = []

        for chunk in chunks:
            with tempfile.TemporaryDirectory() as d:
                if full_generation:
                    f = os.path.join(d, "history_prompt.npz")
                    bark.save_as_prompt(f, full_generation)
                    history_prompt = f
                full_generation, audio_array = bark.generate_audio(
                    chunk,
                    history_prompt=history_prompt,
                    text_temp=self.text_temp,
                    waveform_temp=self.waveform_temp,
                    output_full=True,
                )
                audio_chunks.append(audio_array)

        audio_array = np.concatenate(audio_chunks)

        return audio_array


class ElevenLabsTTS(BaseTTS):
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Args:
            voice: voice in which audio is generated.
        """

        super().__init__()

        self.api_key = api_key

    def generate_audio(self, text: str, voice: Optional[str] = None) -> None:
        """
        Args:
            text: text to be turned into audio.
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
