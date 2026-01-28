"""SharePoint Reader Events and Types for LlamaIndex."""

from enum import Enum


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
