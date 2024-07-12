import inspect
import asyncio
from functools import wraps
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

        if get_origin(return_hint) == Union:
            return_types = list(get_args(return_hint))
            return_types = [t for t in return_types if t != type(None)]
        elif not isinstance(return_hint, list):
            return_types = [return_hint]
        else:
            return_types = return_hint

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


def service(name: str = None):
    """Decorator used to mark methods as services."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            # Measure execution time
            start_time = asyncio.get_event_loop().time()
            result = await func(self, *args, **kwargs)
            end_time = asyncio.get_event_loop().time()

            # cumulate average execution time
            if wrapper.avg_time is None:
                wrapper.avg_time = end_time - start_time
            else:
                wrapper.avg_time = (wrapper.avg_time + end_time - start_time) / 2.0

            return result

        # Mark as a service
        wrapper.__is_service__ = True
        wrapper.__service_name__ = name or func.__name__
        wrapper.avg_time = None

        return wrapper

    return decorator
