import inspect
from typing import (
    get_args,
    get_origin,
    Any,
    List,
    Optional,
    Union,
    Callable,
    get_type_hints,
)

from llama_index.core.workflow.events import Event

from .errors import WorkflowValidationError


def validate_step_signature(fn: Callable) -> None:
    """Given a function, ensure the signature is compatible with a workflow step.

    Two types of signatures are supported:
        - self, ev: Event, for class methods
        - ev: Event, for free functions
    """
    sig = inspect.signature(fn)

    # At least one parameter
    if len(sig.parameters) == 0:
        msg = "Step signature must have at least one parameter"
        raise WorkflowValidationError(msg)

    # All parameters must be annotated
    num_of_possible_events = 0
    for name, t in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        if get_origin(t.annotation) in (Union, Optional):
            event_types = get_args(t.annotation)
        else:
            event_types = [t.annotation]

        all_events = all(et == Event or issubclass(et, Event) for et in event_types)

        if not all_events:
            msg = "Step signature parameters must be annotated with an Event type"
            raise WorkflowValidationError(msg)

        num_of_possible_events += 1

    # Number of events in the signature must be exactly one
    if num_of_possible_events > 1:
        msg = "Step signature must contain exactly one parameter of type Event and no other parameters"
        raise WorkflowValidationError(msg)


def get_steps_from_class(_class: object) -> dict:
    """Given a class, return the list of its methods that were defined as steps."""
    step_methods = {}
    all_methods = inspect.getmembers(_class, predicate=inspect.ismethod)

    for name, method in all_methods:
        if hasattr(method, "__step_config"):
            step_methods[name] = method

    return step_methods


def get_param_types(param: inspect.Parameter) -> List[object]:
    """Extract the types of a parameter. Handles Union and Optional types."""
    typ = param.annotation
    if typ is inspect.Parameter.empty:
        return [Any]
    if get_origin(typ) in (Union, Optional):
        return [t for t in get_args(typ) if t is not type(None)]
    return [typ]


def get_return_types(func: Callable) -> List[object]:
    """Extract the return type hints from a function.

    Handles Union, Optional, and List types.
    """
    type_hints = get_type_hints(func)
    return_hint = type_hints.get("return", [Any])

    origin = get_origin(return_hint)

    if origin == Union:
        # Optional is Union[type, None] so it's covered here
        return [t for t in get_args(return_hint) if t is not type(None)]
    else:
        return [return_hint]
