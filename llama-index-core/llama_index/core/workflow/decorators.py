from .utils import get_events_from_signature


def step(fn):
    """Decorator used to mark methods and functions as workflow steps.

    Decorators are evaluated at import time, but we need to wait for
    starting the communication channels until runtime. For this reason,
    we temporarily store the list of events that will be consumed by this
    step in the function object itself.
    """
    events = get_events_from_signature(fn)
    if not events:
        raise ValueError("The method must receive at least one Event")

    fn.__target_events = events
    return fn
