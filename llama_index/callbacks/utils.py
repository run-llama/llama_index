from typing import Any, Callable, cast
from llama_index.callbacks.base import CallbackManager

def trace_method(
    trace_id: str, callback_manager_attr: str = "callback_manager"
) -> Callable[[Callable], Callable]:
    """
    Decorator to trace a method.

    Example:
        @trace_method("my_trace_id")
        def my_method(self):
            pass
    
    Assumes that the self instance has a CallbackManager instance in an attribute named `callback_manager`.
    This can be overridden by passing in a `callback_manager_attr` keyword argument.
    """
    
    def decorator(func: Callable) -> Callable:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            callback_manager = getattr(self, callback_manager_attr)
            callback_manager = cast(CallbackManager, callback_manager)
            with callback_manager.as_trace(trace_id):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
