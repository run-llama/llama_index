"""Text to speech module."""

from typing import Optional
import tempfile
import os
import numpy as np


class BaseTTS:
    def __init__(self) -> None:
        pass


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

        import_err_msg = "`bark` package not found, \
            please run `pip install git+https://github.com/suno-ai/bark.git`"
        try:
            import bark
        except ImportError:
            raise ImportError(import_err_msg)

        self.generate_audio_fn = bark.generate_audio
        self.save_as_prompt = bark.save_as_prompt
        self.text_temp = text_temp
        self.waveform_temp = waveform_temp
        self.lang_speaker_voice = lang_speaker_voice

    def generate_bark_audio(self, text: str) -> None:
        """
        Args:
            text: text to be turned into audio.
        """

        # text to be chunked into 10 words each to avoid hallicunation
        words = text.split()
        chunks = [words[i : i + 10] for i in range(0, len(words), 10)]
        chunks = [" ".join(chunk) for chunk in chunks]

        full_generation = None
        history_prompt = self.lang_speaker_voice
        audio_chunks = []

        for chunk in chunks:
            with tempfile.TemporaryDirectory() as d:
                if full_generation:
                    f = os.path.join(d, "history_prompt.npz")
                    self.save_as_prompt(f, full_generation)
                    history_prompt = f
                full_generation, audio_array = self.generate_audio_fn(
                    chunk,
                    history_prompt=history_prompt,
                    text_temp=self.text_temp,
                    waveform_temp=self.waveform_temp,
                    output_full=True,
                )
                audio_chunks.append(audio_array)

        audio_array = np.concatenate(audio_chunks)

        return audio_array
