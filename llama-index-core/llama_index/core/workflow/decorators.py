from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from llama_index.core.bridge.pydantic import BaseModel, ConfigDict

from .errors import WorkflowValidationError
from .utils import (
    is_free_function,
    validate_step_signature,
    inspect_signature,
    ServiceDefinition,
)

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow
from .retry_policy import RetryPolicy


class StepConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    accepted_events: List[Any]
    event_name: str
    return_types: List[Any]
    context_parameter: Optional[str]
    num_workers: int
    requested_services: List[ServiceDefinition]
    retry_policy: Optional[RetryPolicy]


def step(
    *args: Any,
    workflow: Optional[Type["Workflow"]] = None,
    pass_context: bool = False,
    num_workers: int = 1,
    retry_policy: Optional[RetryPolicy] = None,
) -> Callable:
    """Decorator used to mark methods and functions as workflow steps.

    Decorators are evaluated at import time, but we need to wait for
    starting the communication channels until runtime. For this reason,
    we temporarily store the list of events that will be consumed by this
    step in the function object itself.
    """

    def decorator(func: Callable) -> Callable:
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise WorkflowValidationError(
                "num_workers must be an integer greater than 0"
            )

        # This will raise providing a message with the specific validation failure
        spec = inspect_signature(func)
        validate_step_signature(spec)
        event_name, accepted_events = next(iter(spec.accepted_events.items()))

        # store the configuration in the function object
        func.__step_config = StepConfig(  # type: ignore[attr-defined]
            accepted_events=accepted_events,
            event_name=event_name,
            return_types=spec.return_types,
            context_parameter=spec.context_parameter,
            num_workers=num_workers,
            requested_services=spec.requested_services or [],
            retry_policy=retry_policy,
        )

        # If this is a free function, call add_step() explicitly.
        if is_free_function(func.__qualname__):
            if workflow is None:
                msg = f"To decorate {func.__name__} please pass a workflow class to the @step decorator."
                raise WorkflowValidationError(msg)
            workflow.add_step(func)

        return func

    if len(args):
        # The decorator was used without parentheses, like `@step`
        func = args[0]
        decorator(func)
        return func
    return decorator
