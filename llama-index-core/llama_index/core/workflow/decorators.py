from .utils import valid_step_signature


def step(*events):
    """Decorator used to mark methods and functions as workflow steps.

    Decorators are evaluated at import time, but we need to wait for
    starting the communication channels until runtime. For this reason,
    we temporarily store the list of events that will be consumed by this
    step in the function object itself.
    """

    def decorator(fn):
        if not valid_step_signature(fn):
            msg = "Wrong signature for step function: must be either (self, *args) or (*args)"
            raise ValueError(msg)
        fn.__target_events = events
        return fn

    return decorator
