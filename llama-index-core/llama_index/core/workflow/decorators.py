import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
)

from llama_index.core.bridge.pydantic import BaseModel
from .utils import validate_step_signature, get_param_types, get_return_types


class StepConfig(BaseModel):
    accepted_events: List[Any]
    event_name: str
    return_types: List[Any]
    kwargs: Dict[str, Any]


def step(**kwargs: Any):
    """Decorator used to mark methods and functions as workflow steps.

    Decorators are evaluated at import time, but we need to wait for
    starting the communication channels until runtime. For this reason,
    we temporarily store the list of events that will be consumed by this
    step in the function object itself.
    """

    def decorator(func: Callable) -> Callable:
        # This will raise providing a message with the specific validation failure
        validate_step_signature(func)

        # Get the function signature
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            event_types = get_param_types(param)
            event_name = name

        # Extract return type
        return_types = get_return_types(func)

        # store the configuration in the function object
        func.__step_config = StepConfig(
            accepted_events=event_types,
            event_name=event_name,
            return_types=return_types,
            kwargs=kwargs,
        )

        return func

    return decorator
