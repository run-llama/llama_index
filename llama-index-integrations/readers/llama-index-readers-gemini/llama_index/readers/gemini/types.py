"""Data models and type definitions for Gemini PDF Reader."""

import logging
import time
from io import BufferedIOBase
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type definitions
FileInput = Union[str, bytes, BufferedIOBase, Path]


class TextChunk(BaseModel):
    """A text chunk with metadata extracted from document."""

    text: str = Field(description="The extracted text content")


class Chunks(BaseModel):
    """Model for structured output from Gemini."""

    chunks: List[TextChunk] = Field(description="List of text chunks")


class ProcessingStats(BaseModel):
    """Statistics about the PDF processing operation."""

    start_time: float
    end_time: Optional[float] = None
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get the processing duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def seconds_per_page(self) -> float:
        """Get the processing rate in pages per second."""
        if self.processed_pages == 0:
            return 0
        return self.duration / self.processed_pages

    @property
    def chunks_per_page(self) -> float:
        """Get the average number of chunks per page."""
        if self.processed_pages == 0:
            return 0
        return self.total_chunks / self.processed_pages
