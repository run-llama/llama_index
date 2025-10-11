from workflows.decorators import StepConfig  # noqa
from workflows.decorators import step as upstream_step  # noqa

from typing import Callable, Any


def step(*args: Any, **kwargs: Any) -> Callable:
    # Remove old, unused parameter
    kwargs.pop("pass_context", None)
    return upstream_step(*args, **kwargs)
