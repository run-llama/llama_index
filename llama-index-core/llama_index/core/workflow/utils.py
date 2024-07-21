import inspect

from llama_index.core.workflow.events import Event


def valid_step_signature(fn) -> bool:
    """Given a function, ensure the signature is compatible with a workflow step.

    Two types of signatures are supported:
        - self, *args, for class methods
        - *args, for free functions
    """
    sig = inspect.signature(fn)

    # try to bind as many events as the function accepts
    try:
        number_of_events = len(sig.parameters)
        if number_of_events == 0:
            return False

        sig.bind(*[Event() for _ in range(number_of_events)])
    except TypeError:
        try:
            number_of_events = len(sig.parameters) - 1
            sig.bind(object(), *[Event() for _ in range(number_of_events)])
        except TypeError:
            return False

    return True


def get_steps_from_class(_class: object) -> dict:
    """Given a class, return the list of its methods that were defined as steps."""
    step_methods = {}
    all_methods = inspect.getmembers(_class, predicate=inspect.ismethod)

    for name, method in all_methods:
        if hasattr(method, "__step_config"):
            step_methods[name] = method

    return step_methods
