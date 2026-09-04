import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FunASRReader(BaseReader):
    """Reader for local FunASR speech-to-text transcription."""

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        recognizer: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        remove_tags: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.generate_kwargs = generate_kwargs or {}
        self.remove_tags = remove_tags

        if recognizer is not None:
            self.recognizer = recognizer
        else:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "Please install FunASR with `pip install funasr` "
                    "to use FunASRReader."
                ) from exc

            self.recognizer = AutoModel(
                model=self.model,
                device=self.device,
                **self.model_kwargs,
            )

    def _clean_text(self, text: str) -> str:
        """Clean SenseVoice-style tags from transcript text."""
        if self.remove_tags:
            text = re.sub(r"<\|.*?\|>", "", text)

        return text.strip()

    def _extract_text(self, result: Any) -> str:
        """Extract transcript text from common FunASR result formats."""
        if isinstance(result, list) and result:
            result = result[0]

        if isinstance(result, dict):
            text = (
                result.get("text")
                or result.get("transcript")
                or result.get("transcription")
            )
            if text:
                return self._clean_text(text)

            segments = result.get("segments")
            if isinstance(segments, list):
                segment_text = " ".join(
                    segment.get("text", "")
                    for segment in segments
                    if isinstance(segment, dict)
                )
                return self._clean_text(segment_text)

        return self._clean_text(str(result)) if result else ""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Transcribe an audio file using local FunASR."""
        if fs is not None:
            raise NotImplementedError(
                "FunASRReader currently supports local file paths only."
            )

        file_path = Path(file)

        result = self.recognizer.generate(
            input=str(file_path),
            **self.generate_kwargs,
        )

        transcript = self._extract_text(result)

        metadata = extra_info.copy() if extra_info else {}
        metadata.update(
            {
                "source_path": str(file_path),
                "model": self.model,
                "device": self.device,
            }
        )

        return [Document(text=transcript, metadata=metadata)]