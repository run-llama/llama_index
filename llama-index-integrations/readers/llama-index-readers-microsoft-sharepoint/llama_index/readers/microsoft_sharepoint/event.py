"""SharePoint Reader Events and Types for LlamaIndex."""

from enum import Enum
from typing import Any, Optional

from llama_index.core.instrumentation.events import BaseEvent


class FileType(Enum):
    """Enum for file types supported by custom parsers."""

    IMAGE = "image"
    DOCUMENT = "document"
    TEXT = "text"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "md"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    PDF = "pdf"
    JSON = "json"
    TXT = "txt"
    UNKNOWN = "unknown"


class TotalPagesToProcessEvent(BaseEvent):
    """Event emitted when the total number of pages to process is known."""

    total_pages: int

    @classmethod
    def class_name(cls) -> str:
        return "TotalPagesToProcessEvent"


class PageDataFetchStartedEvent(BaseEvent):
    """Event emitted when fetching data for a page starts."""

    page_id: str

    @classmethod
    def class_name(cls) -> str:
        return "PageDataFetchStartedEvent"


class PageDataFetchCompletedEvent(BaseEvent):
    """Event emitted when fetching data for a page completes."""

    page_id: str
    document: Optional[Any] = None

    @classmethod
    def class_name(cls) -> str:
        return "PageDataFetchCompletedEvent"


class PageSkippedEvent(BaseEvent):
    """Event emitted when a page is skipped (e.g., filtered out by callback)."""

    page_id: str

    @classmethod
    def class_name(cls) -> str:
        return "PageSkippedEvent"


class PageFailedEvent(BaseEvent):
    """Event emitted when processing a page fails."""

    page_id: str
    error: str

    @classmethod
    def class_name(cls) -> str:
        return "PageFailedEvent"
