import json
import logging
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse

logger = logging.getLogger("uvicorn")


class VercelStreamResponse(StreamingResponse):
    """
    Converts preprocessed events into Vercel-compatible streaming response format.
    """

    TEXT_PREFIX = "0:"
    DATA_PREFIX = "8:"
    ERROR_PREFIX = "3:"

    def __init__(
        self,
        content_stream: AsyncGenerator[str, None],
        *args,
        **kwargs,
    ):
        super().__init__(content=content_stream, *args, **kwargs)

    @classmethod
    def convert_text(cls, token: str) -> str:
        """Convert text event to Vercel format."""
        # Escape newlines and double quotes to avoid breaking the stream
        token = json.dumps(token)
        return f"{cls.TEXT_PREFIX}{token}\n"

    @classmethod
    def convert_data(cls, data: dict | str) -> str:
        """Convert data event to Vercel format."""
        data_str = json.dumps(data) if isinstance(data, dict) else data
        return f"{cls.DATA_PREFIX}[{data_str}]\n"

    @classmethod
    def convert_error(cls, error: str) -> str:
        """Convert error event to Vercel format."""
        error_str = json.dumps(error)
        return f"{cls.ERROR_PREFIX}{error_str}\n"
