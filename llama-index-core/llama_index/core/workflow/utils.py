import inspect
from importlib import import_module
from typing import (
    get_args,
    get_origin,
    Any,
    List,
    Optional,
    Union,
    Callable,
    Dict,
    get_type_hints,
)

# handle python version compatibility
try:
    from types import UnionType  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing import Union as UnionType

from llama_index.core.bridge.pydantic import BaseModel, ConfigDict

from .events import Event, EventType
from .errors import WorkflowValidationError


class ServiceDefinition(BaseModel):
    # Make the service definition hashable
    model_config = ConfigDict(frozen=True)

    name: str
    service: Any
    default_value: Optional[Any]


class StepSignatureSpec(BaseModel):
    """A Pydantic model representing the signature of a step function or method."""

    accepted_events: Dict[str, List[EventType]]
    return_types: List[Any]
    context_parameter: Optional[str]
    requested_services: Optional[List[ServiceDefinition]]


def inspect_signature(fn: Callable) -> StepSignatureSpec:
    """Given a function, ensure the signature is compatible with a workflow step."""
    sig = inspect.signature(fn)

    accepted_events: Dict[str, List[EventType]] = {}
    context_parameter = None
    requested_services = []

    # Inspect function parameters
    for name, t in sig.parameters.items():
        # Ignore self and cls
        if name in ("self", "cls"):
            continue

        # Get name and type of the Context param
        if hasattr(t.annotation, "__name__") and t.annotation.__name__ == "Context":
            context_parameter = name
            continue

        # Collect name and types of the event param
        param_types = _get_param_types(t)
        if all(
            param_t == Event
            or (inspect.isclass(param_t) and issubclass(param_t, Event))
            for param_t in param_types
        ):
            accepted_events[name] = param_types
            continue

        # Everything else will be treated as a service request
        default_value = t.default
        if default_value is inspect.Parameter.empty:
            default_value = None

        requested_services.append(
            ServiceDefinition(
                name=name, service=param_types[0], default_value=default_value
            )
        )

    return StepSignatureSpec(
        accepted_events=accepted_events,
        return_types=_get_return_types(fn),
        context_parameter=context_parameter,
        requested_services=requested_services,
    )


def validate_step_signature(spec: StepSignatureSpec) -> None:
    num_of_events = len(spec.accepted_events)
    if num_of_events == 0:
        msg = "Step signature must have at least one parameter annotated as type Event"
        raise WorkflowValidationError(msg)
    elif num_of_events > 1:
        msg = f"Step signature must contain exactly one parameter of type Event but found {num_of_events}."
        raise WorkflowValidationError(msg)

    if not spec.return_types:
        msg = f"Return types of workflows step functions must be annotated with their type."
        raise WorkflowValidationError(msg)


def get_steps_from_class(_class: object) -> Dict[str, Callable]:
    """Given a class, return the list of its methods that were defined as steps."""
    step_methods = {}
    all_methods = inspect.getmembers(_class, predicate=inspect.isfunction)

    for name, method in all_methods:
        if hasattr(method, "__step_config"):
            step_methods[name] = method

    return step_methods


def get_steps_from_instance(workflow: object) -> Dict[str, Callable]:
    """Given a workflow instance, return the list of its methods that were defined as steps."""
    step_methods = {}
    all_methods = inspect.getmembers(workflow, predicate=inspect.ismethod)

    for name, method in all_methods:
        if hasattr(method, "__step_config"):
            step_methods[name] = method

    return step_methods


def _get_param_types(param: inspect.Parameter) -> List[Any]:
    """Extract the types of a parameter. Handles Union and Optional types."""
    typ = param.annotation
    if typ is inspect.Parameter.empty:
        return [Any]
    if get_origin(typ) in (Union, Optional, UnionType):
        return [t for t in get_args(typ) if t is not type(None)]
    return [typ]


def _get_return_types(func: Callable) -> List[Any]:
    """Extract the return type hints from a function.

    Handles Union, Optional, and List types.
    """
    type_hints = get_type_hints(func)
    return_hint = type_hints.get("return")
    if return_hint is None:
        return []

    origin = get_origin(return_hint)
    if origin in (Union, UnionType):
        # Optional is Union[type, None] so it's covered here
        return [t for t in get_args(return_hint) if t is not type(None)]
    else:
        return [return_hint]


def is_free_function(qualname: str) -> bool:
    """Determines whether a certain qualified name points to a free function.

    The strategy should be able to spot nested functions, for details see PEP-3155.
    """
    if not qualname:
        msg = "The qualified name cannot be empty"
        raise ValueError(msg)

    toks = qualname.split(".")
    if len(toks) == 1:
        # e.g. `my_function`
        return True
    elif "<locals>" not in toks:
        # e.g. `MyClass.my_method`
        return False
    else:
        return toks[-2] == "<locals>"


def get_qualified_name(value: Any) -> str:
    """Get the qualified name of a value."""
    return value.__module__ + "." + value.__class__.__name__


def import_module_from_qualified_name(qualified_name: str) -> Any:
    """Import a module from a qualified name."""
    module_path = qualified_name.rsplit(".", 1)
    module = import_module(module_path[0])
    return getattr(module, module_path[1])
