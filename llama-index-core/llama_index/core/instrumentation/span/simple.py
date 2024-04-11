from typing import Dict, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.span.base import BaseSpan
from datetime import datetime


class SimpleSpan(BaseSpan):
    """Simple span class."""

    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: Optional[datetime] = Field(default=None)
    duration: float = Field(default=float, description="Duration of span in seconds.")
    metadata: Optional[Dict] = Field(default=None)
