"""FunASR audio reader."""

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

_TARGET_SR = 16000
_SENSE_VOICE_LANGUAGES = {"auto", "zh", "en", "yue", "ja", "ko"}


class FunASRReader(BaseReader):
    """
    FunASR reader.

    Reads audio files and transcribes them locally with FunASR
    (SenseVoice / Paraformer / Fun-ASR-Nano) - no cloud API. Strong on Chinese
    and 50+ languages.

    Args:
        model (str): FunASR model id. Defaults to "iic/SenseVoiceSmall".
        device (str): Inference device, e.g. "cpu" or "cuda:0".
        hub (str): Model hub, "ms" (ModelScope) or "hf" (HuggingFace).
        language (str): SenseVoice decoding language ("auto"/"zh"/"en"/...).
        use_itn (bool): Apply inverse text normalization.
        vad_model (Optional[str]): VAD model; enables arbitrary-length audio.
        batch_size_s (int): Dynamic batch size (seconds) for VAD segments.

    """

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        *,
        device: str = "cpu",
        hub: str = "ms",
        language: str = "auto",
        use_itn: bool = True,
        vad_model: Optional[str] = "fsmn-vad",
        batch_size_s: int = 300,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.model_name = model
        self.device = device
        self.hub = hub
        self.language = language
        self.use_itn = use_itn
        self.vad_model = vad_model
        self.batch_size_s = batch_size_s
        self._model = None
        self._postprocess = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from funasr import AutoModel
                from funasr.utils.postprocess_utils import (
                    rich_transcription_postprocess,
                )
            except ImportError as e:
                raise ImportError(
                    "llama-index-readers-funasr requires funasr. "
                    "Install with `pip install funasr`."
                ) from e
            self._postprocess = rich_transcription_postprocess
            kwargs = {
                "model": self.model_name,
                "hub": self.hub,
                "device": self.device,
                "disable_update": True,
            }
            if self.vad_model:
                kwargs["vad_model"] = self.vad_model
                kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
            self._model = AutoModel(**kwargs)
        return self._model

    def _transcribe(self, input_file: Union[str, Path, bytes]) -> str:
        try:
            import librosa
        except ImportError as e:
            raise ImportError(
                "llama-index-readers-funasr requires librosa for audio decoding. "
                "Install with `pip install librosa`."
            ) from e

        model = self._ensure_model()
        source: Union[str, BytesIO]
        source = BytesIO(input_file) if isinstance(input_file, bytes) else str(input_file)
        audio, _ = librosa.load(source, sr=_TARGET_SR, mono=True)

        gen_kwargs = {
            "input": audio,
            "cache": {},
            "use_itn": self.use_itn,
            "batch_size_s": self.batch_size_s,
        }
        if "SenseVoice" in self.model_name:
            lang = self.language if self.language in _SENSE_VOICE_LANGUAGES else "auto"
            gen_kwargs["language"] = lang

        result = model.generate(**gen_kwargs)
        text = result[0]["text"] if result else ""
        return self._postprocess(text).strip()

    def load_data(
        self,
        input_file: Union[str, Path, bytes],
        extra_info: Optional[Dict] = None,
        **transcribe_kwargs: dict,
    ) -> List[Document]:
        """Transcribe an audio file into a list with a single Document."""
        text = self._transcribe(input_file)
        metadata = extra_info or {}
        return [Document(text=text, metadata=metadata)]
