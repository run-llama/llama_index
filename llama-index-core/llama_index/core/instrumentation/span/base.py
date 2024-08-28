from typing import Any, Dict, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict


class BaseSpan(BaseModel):
    """Base data class representing a span."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    id_: str = Field(default_factory=str, description="Id of span.")
    parent_id: Optional[str] = Field(default=None, description="Id of parent span.")
    tags: Dict[str, Any] = Field(default={})
