from typing import List, Optional, Type
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.event_management.handlers import BaseEventHandler
from llama_index.core.event_management.events.base import BaseEvent


class Dispatcher(BaseModel):
    name: str = Field(default_factory=str, description="Name of dispatcher")
    handlers: List[BaseEventHandler] = Field(
        default=[], description="List of attached handlers"
    )
    parent: Optional["Dispatcher"] = Field(
        default_factory=None, description="List of parent dispatchers"
    )
    propagate: bool = Field(
        default=True,
        description="Whether to propagate the event to parent dispatchers and their handlers",
    )

    def add_handler(self, handler) -> None:
        """Add handler to set of handlers."""
        self.handlers += [handler]

    def dispatch(self, event_cls: Type[BaseEvent]) -> None:
        """Dispatch event to all registered handlers."""
        event = event_cls()
        for h in self.handlers:
            h.handle(event)

    @property
    def log_name(self) -> str:
        """Name to be used in logging."""
        if self.parent:
            return f"{self.parent.name}.{self.name}"
        else:
            return self.name

    class Config:
        arbitrary_types_allowed = True
