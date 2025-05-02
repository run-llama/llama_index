from typing import Any, Dict, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4
from datetime import datetime

from llama_index.core.instrumentation.span import active_span_id


class BaseEvent(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # copy_on_model_validation = "deep"  # not supported in Pydantic V2...
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    id_: str = Field(default_factory=lambda: str(uuid4()))
    span_id: Optional[str] = Field(default_factory=active_span_id.get)  # type: ignore
    tags: Dict[str, Any] = Field(default={})

    @classmethod
    def class_name(cls) -> str:
        """Return class name."""
        return "BaseEvent"

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Keep for backwards compatibility."""
        return self.model_dump(**kwargs)

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["class_name"] = self.class_name()
        return data
