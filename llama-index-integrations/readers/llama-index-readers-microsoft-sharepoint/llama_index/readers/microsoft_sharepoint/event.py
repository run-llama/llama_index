from enum import Enum
from llama_index.core.schema import Document
from llama_index.core.instrumentation.events.base import BaseEvent


class FileType(Enum):
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


# LlamaIndex instrumentation events
class TotalPagesToProcessEvent(BaseEvent):
    """Event emitted when the total number of pages to process is determined."""

    total_pages: int

    @classmethod
    def class_name(cls) -> str:
        return "TotalPagesToProcessEvent"


class PageDataFetchStartedEvent(BaseEvent):
    """Event emitted when processing of a page begins."""

    page_id: str

    @classmethod
    def class_name(cls) -> str:
        return "PageDataFetchStartedEvent"


class PageDataFetchCompletedEvent(BaseEvent):
    """Event emitted when a page is successfully processed."""

    page_id: str
    document: Document

    @classmethod
    def class_name(cls) -> str:
        return "PageDataFetchCompletedEvent"


class PageSkippedEvent(BaseEvent):
    """Event emitted when a page is skipped due to callback decision."""

    page_id: str

    @classmethod
    def class_name(cls) -> str:
        return "PageSkippedEvent"


class PageFailedEvent(BaseEvent):
    """Event emitted when page processing fails."""

    page_id: str
    error: str

    @classmethod
    def class_name(cls) -> str:
        return "PageFailedEvent"
