from __future__ import annotations

from io import IOBase
from typing import List, Literal, Optional, Tuple

import asyncio

import google.genai.types as types
from google.genai import Client


class FileManager:
    """Manage file parts and File API lifecycle."""

    def __init__(
        self,
        *,
        file_mode: Literal["inline", "fileapi", "hybrid"],
        client: Client,
    ) -> None:
        self._file_mode = file_mode
        self._client = client

    @property
    def file_mode(self) -> Literal["inline", "fileapi", "hybrid"]:
        return self._file_mode

    async def create_part(
        self, file_buffer: IOBase, mime_type: str
    ) -> Tuple[types.Part, Optional[str]]:
        """Create a Part for the given file, uploading when required."""
        if self._file_mode in ("inline", "hybrid"):
            file_buffer.seek(0, 2)
            size = file_buffer.tell()
            file_buffer.seek(0)

            if size < 20 * 1024 * 1024:
                return (
                    types.Part.from_bytes(data=file_buffer.read(), mime_type=mime_type),
                    None,
                )
            if self._file_mode == "inline":
                raise ValueError("Files in inline mode must be smaller than 20MB.")

        uploaded = await self._client.aio.files.upload(
            file=file_buffer,
            config=types.UploadFileConfig(mime_type=mime_type),
        )

        while uploaded.state and uploaded.state.name == "PROCESSING":
            await asyncio.sleep(2)
            uploaded = await self._client.aio.files.get(name=uploaded.name)

        if uploaded.state and uploaded.state.name == "FAILED":
            raise ValueError("Failed to upload the file with FileAPI")

        return (
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type),
            uploaded.name,
        )

    def cleanup(self, file_api_names: List[str]) -> None:
        """Delete uploaded files created via File API (sync)."""
        for name in file_api_names:
            self._client.files.delete(name=name)

    async def acleanup(self, file_api_names: List[str]) -> None:
        """Delete uploaded files created via File API (async)."""
        await asyncio.gather(
            *[self._client.aio.files.delete(name=name) for name in file_api_names]
        )
