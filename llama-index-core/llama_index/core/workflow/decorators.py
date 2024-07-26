import inspect
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from llama_index.core.bridge.pydantic import BaseModel
from .utils import validate_step_signature, get_param_types, get_return_types
from .errors import WorkflowValidationError


if TYPE_CHECKING:
    from .workflow import Workflow


class StepConfig(BaseModel):
    accepted_events: List[Any]
    event_name: str
    return_types: List[Any]


def step(workflow: Optional["Workflow"] = None):
    """Decorator used to mark methods and functions as workflow steps.

    Decorators are evaluated at import time, but we need to wait for
    starting the communication channels until runtime. For this reason,
    we temporarily store the list of events that will be consumed by this
    step in the function object itself.
    """

    def decorator(func: Callable) -> Callable:
        # This will raise providing a message with the specific validation failure
        validate_step_signature(func)

        # Determine if this is a free function
        name_toks = func.__qualname__.split(".")
        is_free_func = len(name_toks) > 1 and name_toks[-2] == "<locals>"

        # If this is a free function, add it to the workflow instance
        if is_free_func:
            if workflow is None:
                msg = f"To decorate {func.__name__} please pass a workflow instance to the @step() decorator."
                raise WorkflowValidationError(msg)
            workflow.add_step(func)

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
        )

        return func

    return decorator
