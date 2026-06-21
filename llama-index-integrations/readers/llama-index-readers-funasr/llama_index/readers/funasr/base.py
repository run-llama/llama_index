"""FunASR reader."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FunASRReader(BaseReader):
    """
    FunASR reader.

    Reads audio files and transcribes them locally with
    [FunASR](https://github.com/modelscope/FunASR) (SenseVoice / Paraformer /
    Fun-ASR-Nano) — multilingual ASR (Chinese, Cantonese, English, Japanese,
    Korean and more) that runs on CPU or GPU with no API key. SenseVoice (the
    default) auto-detects the spoken language, and a built-in FSMN-VAD handles
    long audio.

    Args:
        model (str): FunASR model id on ModelScope/Hugging Face.
            Defaults to ``"iic/SenseVoiceSmall"``.
        device (str): Inference device, ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
        language (str): Spoken language; ``"auto"`` auto-detects. Defaults to ``"auto"``.
        use_itn (bool): Apply inverse text normalization. Defaults to ``True``.

    """

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        language: str = "auto",
        use_itn: bool = True,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "`funasr` package not found, please run `pip install funasr`"
            )
        self._language = language
        self._use_itn = use_itn
        self._model = AutoModel(
            model=model,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )

    def load_data(
        self,
        input_file: Union[str, Path],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Transcribe ``input_file`` and return a single ``Document`` with the text."""
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        result = self._model.generate(
            input=str(input_file),
            cache={},
            language=self._language,
            use_itn=self._use_itn,
        )
        text = rich_transcription_postprocess(result[0]["text"]).strip() if result else ""
        metadata = {"file_name": str(input_file), **(extra_info or {})}
        return [Document(text=text, metadata=metadata)]
