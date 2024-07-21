import inspect
from typing import (
    get_args,
    get_origin,
    get_type_hints,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Type,
)

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.workflow.utils import valid_step_signature


class StepConfig(BaseModel):
    required_events: Dict[str, List[Any]]
    optional_events: Dict[str, List[Any]]
    return_types: List[Any]
    consume_all: bool
    kwargs: Dict[str, Any]


def get_param_types(param: inspect.Parameter) -> List[Type]:
    """Extract the types of a parameter. Handles Union and Optional types."""
    typ = param.annotation
    if typ is inspect.Parameter.empty:
        return [Any]
    if get_origin(typ) in (Union, Optional):
        return [t for t in get_args(typ) if t is not type(None)]
    return [typ]


def get_return_types(return_hint: Any) -> List[Type]:
    """Extract the types of a return hint. Handles Union, Optional, and List types."""
    if get_origin(return_hint) == Union:
        return [t for t in get_args(return_hint) if t is not type(None)]
    if get_origin(return_hint) == Optional:
        return [get_args(return_hint)[0]]
    if get_origin(return_hint) == list:
        return get_args(return_hint)
    if not isinstance(return_hint, list):
        return [return_hint]
    return return_hint


def step(consume_all: bool = False, **kwargs: Any):
    """Decorator used to mark methods and functions as workflow steps.

    Decorators are evaluated at import time, but we need to wait for
    starting the communication channels until runtime. For this reason,
    we temporarily store the list of events that will be consumed by this
    step in the function object itself.
    """

    def decorator(func: Callable) -> Callable:
        if not valid_step_signature(func):
            msg = "Wrong signature for step function: must be either (self, *args) or (*args)"
            raise ValueError(msg)

        # Get type hints for the function
        type_hints = get_type_hints(func)

        # Get the function signature
        sig = inspect.signature(func)

        # Extract parameter types and separate required and optional
        required_params = {}
        optional_params = {}

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            param_types = get_param_types(param)

            if param.default == param.empty:
                required_params[name] = param_types
            else:
                optional_params[name] = param_types

        # Extract return type
        return_hint = type_hints.get("return", [Any])
        return_types = get_return_types(return_hint)

        # store the configuration in the function object
        func.__step_config = StepConfig(
            required_events=required_params,
            optional_events=optional_params,
            return_types=return_types,
            consume_all=consume_all,
            kwargs=kwargs,
        )

        return func

    return decorator
