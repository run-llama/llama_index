from contextlib import contextmanager
from contextvars import ContextVar
from llama_index.core.instrumentation.events.base import BaseEvent

# ContextVar for managing active event tags
active_event_tags = ContextVar("event_tags", default={})


@contextmanager
def event_tags(new_tags):
    token = active_event_tags.set(new_tags)
    try:
        yield
    finally:
        active_event_tags.reset(token)


__all__ = [
    "BaseEvent",
]
