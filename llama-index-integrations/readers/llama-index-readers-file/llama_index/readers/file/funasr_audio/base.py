from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FunASRAudioReader(BaseReader):
    """Audio reader using a FunASR OpenAI-compatible transcription endpoint."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8000",
        model: str = "sensevoice",
        language: Optional[str] = None,
        response_format: str = "json",
        timeout: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model
        self.language = language
        self.response_format = response_format
        self.timeout = timeout

    def _extract_transcript(self, result: Dict[str, Any]) -> str:
        """Extract transcript text from common FunASR-compatible responses."""
        if result.get("text"):
            return result["text"]

        if result.get("transcript"):
            return result["transcript"]

        if result.get("transcription"):
            return result["transcription"]

        segments = result.get("segments")
        if isinstance(segments, list):
            return " ".join(
                segment.get("text", "")
                for segment in segments
                if isinstance(segment, dict)
            ).strip()

        return ""

    def _build_request_data(self) -> Dict[str, str]:
        """Build request payload for the transcription endpoint."""
        data = {
            "model": self.model,
            "response_format": self.response_format,
        }

        if self.language:
            data["language"] = self.language

        return data

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Transcribe an audio file using FunASR endpoint."""
        url = f"{self.endpoint_url}/v1/audio/transcriptions"

        metadata = extra_info.copy() if extra_info else {}
        metadata.update(
            {
                "source_path": str(file),
                "model": self.model,
                "endpoint": url,
            }
        )

        data = self._build_request_data()

        if fs:
            with fs.open(file, "rb") as f:
                response = requests.post(
                    url,
                    files={"file": (Path(file).name, f)},
                    data=data,
                    timeout=self.timeout,
                )
        else:
            with open(file, "rb") as f:
                response = requests.post(
                    url,
                    files={"file": (Path(file).name, f)},
                    data=data,
                    timeout=self.timeout,
                )

        response.raise_for_status()
        result = response.json()

        transcript = self._extract_transcript(result)

        if "segments" in result:
            metadata["segments"] = result["segments"]

        metadata["raw_response"] = result

        return [Document(text=transcript, metadata=metadata)]