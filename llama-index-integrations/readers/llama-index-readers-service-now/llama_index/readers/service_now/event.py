from llama_index.core.instrumentation.events.base import BaseEvent
from typing import Dict, Any
from enum import Enum
from llama_index.core.schema import Document
from pydantic import Field


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
    UNKNOWN = "unknown"


# ServiceNow Knowledge Base Reader Events
# All events use LlamaIndex's standard instrumentation event system
# and inherit from BaseEvent for consistent event handling across the framework
class SNOWKBTotalPagesEvent(BaseEvent):
    """Event fired when total pages to process is determined."""

    total_pages: int = Field(description="Total number of pages to process")


class SNOWKBPageFetchStartEvent(BaseEvent):
    """Event fired when page data fetch starts."""

    page_id: str = Field(description="ID of the page being fetched")


class SNOWKBPageFetchCompletedEvent(BaseEvent):
    """Event fired when page data fetch completes successfully."""

    page_id: str = Field(description="ID of the page that was fetched")
    document: Document = Field(description="The processed document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SNOWKBPageSkippedEvent(BaseEvent):
    """Event fired when a page is skipped."""

    page_id: str = Field(description="ID of the page that was skipped")
    reason: str = Field(description="Reason why the page was skipped")


class SNOWKBPageFailedEvent(BaseEvent):
    """Event fired when page processing fails."""

    page_id: str = Field(description="ID of the page that failed")
    error: str = Field(description="Error message")


class SNOWKBAttachmentProcessingStartEvent(BaseEvent):
    """Event fired when attachment processing starts."""

    page_id: str = Field(description="ID of the parent page")
    attachment_id: str = Field(description="ID of the attachment")
    attachment_name: str = Field(description="Name of the attachment")
    attachment_type: str = Field(description="MIME type of the attachment")
    attachment_size: int = Field(description="Size of the attachment in bytes")
    attachment_link: str = Field(description="Link to the attachment")


class SNOWKBAttachmentProcessedEvent(BaseEvent):
    """Event fired when attachment processing completes successfully."""

    page_id: str = Field(description="ID of the parent page")
    attachment_id: str = Field(description="ID of the attachment")
    attachment_name: str = Field(description="Name of the attachment")
    attachment_type: str = Field(description="MIME type of the attachment")
    attachment_size: int = Field(description="Size of the attachment in bytes")
    attachment_link: str = Field(description="Link to the attachment")


class SNOWKBAttachmentSkippedEvent(BaseEvent):
    """Event fired when an attachment is skipped."""

    page_id: str = Field(description="ID of the parent page")
    attachment_id: str = Field(description="ID of the attachment")
    attachment_name: str = Field(description="Name of the attachment")
    attachment_type: str = Field(description="MIME type of the attachment")
    attachment_size: int = Field(description="Size of the attachment in bytes")
    attachment_link: str = Field(description="Link to the attachment")
    reason: str = Field(description="Reason why the attachment was skipped")


class SNOWKBAttachmentFailedEvent(BaseEvent):
    """Event fired when attachment processing fails."""

    page_id: str = Field(description="ID of the parent page")
    attachment_id: str = Field(description="ID of the attachment")
    attachment_name: str = Field(description="Name of the attachment")
    attachment_type: str = Field(description="MIME type of the attachment")
    attachment_size: int = Field(description="Size of the attachment in bytes")
    attachment_link: str = Field(description="Link to the attachment")
    error: str = Field(description="Error message")
