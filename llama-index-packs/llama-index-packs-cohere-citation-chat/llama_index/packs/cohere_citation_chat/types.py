from typing import List
from dataclasses import dataclass, field, asdict


@dataclass
class Citation:
    """Citation object."""

    text: str
    start: int
    end: int
    document_ids: List[str]

    dict = asdict


@dataclass
class Document:
    """Document object."""

    id: str
    text: str

    dict = asdict


@dataclass
class CitationsSettings:
    """Citations settings."""

    documents_request_param: str = field(default="documents")
    documents_stream_event_type: str = field(default="search-results")
    citations_response_field: str = field(default="citations")
    documents_response_field: str = field(default="documents")
    citations_stream_event_type: str = field(default="citation-generation")

    dict = asdict
