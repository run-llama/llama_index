import inspect

from llama_index.core.workflow.events import Event, EventType


def get_events_from_signature(fn) -> list[EventType]:
    """Given a function, extract the list of Event types accepted by inspecting its signature."""
    events: list[EventType] = []
    sig = inspect.signature(fn)
    for type in sig.parameters.values():
        event_type = type.annotation
        if issubclass(event_type, Event):
            events.append(event_type)
    return events


def get_steps_from_class(_class: object) -> dict:
    """Given a class, return the list of its methods that were defined as steps."""
    step_methods = {}
    all_methods = inspect.getmembers(_class, predicate=inspect.ismethod)

    for name, method in all_methods:
        if hasattr(method, "__target_events"):
            step_methods[name] = method

    return step_methods
