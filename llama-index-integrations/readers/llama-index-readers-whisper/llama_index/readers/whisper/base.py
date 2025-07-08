"""JSON Reader."""

import os
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from typing import Dict, List, Optional, Union
from io import BytesIO

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class WhisperReader(BaseReader):
    """
    Whisper reader.

    Reads audio files and transcribes them using the OpenAI Whisper API.

    Args:
        model (str): The OpenAI Whisper model to use. Defaults to "whisper-1".
        api_key (Optional[str]): The OpenAI API key to use. Uses OPENAI_API_KEY environment variable if not provided.
        client (Optional[OpenAI]): An existing OpenAI client to use.
        async_client (Optional[AsyncOpenAI]): An existing AsyncOpenAI client to use.
        client_kwargs (Optional[dict]): Additional keyword arguments to pass to the OpenAI client.
        transcribe_kwargs (Optional[dict]): Additional keyword arguments to pass to the transcribe method.

    """

    def __init__(
        self,
        model: str = "whisper-1",
        language: str = "en",
        prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        async_client: Optional[AsyncOpenAI] = None,
        client_kwargs: Optional[dict] = None,
        transcribe_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        client_kwargs = client_kwargs or {}
        self.model = model
        self.language = language
        self.prompt = prompt
        self.client = client or OpenAI(api_key=api_key, **client_kwargs)
        self.async_client = async_client or AsyncOpenAI(
            api_key=api_key, **client_kwargs
        )

        self.transcribe_kwargs = transcribe_kwargs or {}

    def _get_default_fs(self) -> LocalFileSystem:
        """Get the default filesystem."""
        return LocalFileSystem()

    def _get_file_path_or_bytes(
        self,
        input_file: Union[str, Path, bytes],
        fs: Optional[AbstractFileSystem] = None,
    ) -> Union[str, BytesIO]:
        """Get the file bytes."""
        fs = fs or self._get_default_fs()

        if isinstance(input_file, (str, Path)):
            abs_path = os.path.abspath(str(input_file))
            if not os.path.exists(abs_path):
                raise ValueError(f"File not found: {abs_path}")

            return abs_path
        elif isinstance(input_file, bytes):
            file_bytes = BytesIO(input_file)
            file_bytes.name = "audio.mp3"
            return file_bytes
        elif hasattr(input_file, "read"):  # File-like object
            return input_file
        else:
            raise ValueError("Invalid input file type")

    def _transcribe(
        self,
        file_path_or_bytes: Union[str, BytesIO],
        transcribe_kwargs: Optional[dict] = None,
    ) -> str:
        """Transcribe the audio file."""
        transcribe_kwargs = transcribe_kwargs or self.transcribe_kwargs

        if isinstance(file_path_or_bytes, str):
            # If it's a file path, open it directly
            with open(file_path_or_bytes, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="text",
                    prompt=self.prompt,
                    **transcribe_kwargs,
                )
        else:
            # Handle BytesIO case (we'll improve this later)
            file_path_or_bytes.seek(0)
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=file_path_or_bytes,
                language=self.language,
                response_format="text",
                prompt=self.prompt,
                **transcribe_kwargs,
            )

        return response

    async def _transcribe_async(
        self,
        file_path_or_bytes: Union[str, BytesIO],
        transcribe_kwargs: Optional[dict] = None,
    ) -> str:
        """Transcribe the audio file asynchronously."""
        transcribe_kwargs = transcribe_kwargs or self.transcribe_kwargs

        if isinstance(file_path_or_bytes, str):
            # If it's a file path, open it directly
            with open(file_path_or_bytes, "rb") as audio_file:
                response = await self.async_client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="text",
                    prompt=self.prompt,
                    **transcribe_kwargs,
                )
        else:
            # Handle BytesIO case (we'll improve this later)
            file_path_or_bytes.seek(0)
            response = await self.async_client.audio.transcriptions.create(
                model=self.model,
                file=file_path_or_bytes,
                language=self.language,
                response_format="text",
                prompt=self.prompt,
                **transcribe_kwargs,
            )

        return response

    def load_data(
        self,
        input_file: Union[str, Path, bytes],
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
        **transcribe_kwargs: dict,
    ) -> List[Document]:
        """Load data from the input file."""
        file_path_or_bytes = self._get_file_path_or_bytes(input_file, fs)

        text = self._transcribe(file_path_or_bytes, transcribe_kwargs)
        metadata = extra_info or {}
        return [Document(text=text, metadata=metadata)]

    async def aload_data(
        self,
        input_file: Union[str, Path, bytes],
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
        **transcribe_kwargs: dict,
    ) -> List[Document]:
        """Load data from the input file asynchronously."""
        file_path_or_bytes = self._get_file_path_or_bytes(input_file, fs)

        text = await self._transcribe_async(file_path_or_bytes, transcribe_kwargs)
        metadata = extra_info or {}
        return [Document(text=text, metadata=metadata)]
