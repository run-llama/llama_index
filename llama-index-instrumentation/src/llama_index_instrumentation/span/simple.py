from datetime import datetime
from typing import Dict, Optional

from pydantic import Field

from .base import BaseSpan


class SimpleSpan(BaseSpan):
    """Simple span class."""

    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: Optional[datetime] = Field(default=None)
    duration: float = Field(default=0.0, description="Duration of span in seconds.")
    metadata: Optional[Dict] = Field(default=None)
