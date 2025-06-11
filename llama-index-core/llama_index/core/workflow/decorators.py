from workflows.decorators import StepConfig  # noqa
from workflows.decorators import step as upstream_step  # noqa

from typing import Callable


def step(*args, **kwargs) -> Callable:
    kwargs.pop("pass_context")
    return upstream_step(*args, **kwargs)
