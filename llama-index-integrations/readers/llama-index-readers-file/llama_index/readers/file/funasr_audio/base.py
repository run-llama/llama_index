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
        response_format: str = "json",
        timeout: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model
        self.response_format = response_format
        self.timeout = timeout

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
                "source": str(file),
                "model": self.model,
                "endpoint_url": self.endpoint_url,
            }
        )

        if fs:
            with fs.open(file, "rb") as f:
                response = requests.post(
                    url,
                    files={"file": (Path(file).name, f)},
                    data={
                        "model": self.model,
                        "response_format": self.response_format,
                    },
                    timeout=self.timeout,
                )
        else:
            with open(file, "rb") as f:
                response = requests.post(
                    url,
                    files={"file": (Path(file).name, f)},
                    data={
                        "model": self.model,
                        "response_format": self.response_format,
                    },
                    timeout=self.timeout,
                )

        response.raise_for_status()
        result = response.json()

        transcript = result.get("text", "")
        metadata["raw_response"] = result

        return [Document(text=transcript, metadata=metadata)]
