from typing import Any, Dict, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field


class BaseSpan(BaseModel):
    """Base data class representing a span."""

    id_: str = Field(default_factory=str, description="Id of span.")
    parent_id: Optional[str] = Field(default=None, description="Id of parent span.")
    tags: Dict[str, Any] = Field(default={})

    class Config:
        arbitrary_types_allowed = True
