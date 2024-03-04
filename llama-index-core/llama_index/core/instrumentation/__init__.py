from llama_index.core.instrumentation.dispatcher import Dispatcher
from llama_index.core.instrumentation.event_handlers import NullEventHandler
from llama_index.core.instrumentation.span_handlers import NullSpanHandler

root_dispatcher: Dispatcher = Dispatcher(
    name="root",
    event_handlers=[NullEventHandler()],
    span_handler=NullSpanHandler(),
    parent=None,
    propagate=False,
)


def get_dispatcher(name: str) -> Dispatcher:
    """Module method that should be used for creating a new Dispatcher."""
    return Dispatcher(name=name, parent=root_dispatcher)
