from typing import Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
from datetime import datetime


class BaseSpan(BaseModel):
    """Base data class representing a span."""

    id_: str = Field(default_factory=str, description="Id of span.")
    parent_id: Optional[str] = Field(default=None, description="Id of parent span.")


class SimpleSpan(BaseSpan):
    """Simple span class."""

    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: Optional[datetime] = Field(default=None)
    duration: int = Field(default=int, description="Duration of span in milliseconds.")
