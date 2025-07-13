from pydantic import BaseModel
from typing import Dict, Any
from enum import Enum
from llama_index.core.schema import Document

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

class EventName(str, Enum):    
    PAGE_DATA_FETCH_STARTED = "page_data_fetch_started"
    PAGE_DATA_FETCH_COMPLETED = "page_data_fetch_completed"
    PAGE_SKIPPED = "page_skipped"
    PAGE_FAILED = "page_failed"

    TOTAL_PAGES_TO_PROCESS = "total_pages_to_process"

    ATTACHMENT_PROCESSING_STARTED = "attachment_processing_started"
    ATTACHMENT_PROCESSED = "attachment_processed"
    ATTACHMENT_SKIPPED = "attachment_skipped"
    ATTACHMENT_FAILED = "attachment_failed"


class Event(BaseModel):
    name: EventName

class AttachmentEvent(Event):
    name: EventName
    attachment_id: str
    page_id: str
    attachment_name: str
    attachment_type: str
    attachment_size: int
    attachment_link: str
    error: str = None

class PageEvent(Event):
    name: EventName
    page_id: str
    document: Document
    error: str = None
    metadata: Dict[str, Any] = {}