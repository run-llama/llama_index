from llama_index.core.instrumentation.events import BaseEvent


class ExceptionEvent(BaseEvent):
    """ExceptionEvent.

    Args:
        exception (BaseException): exception.
    """

    exception: BaseException

    @classmethod
    def class_name(cls):
        """Class name."""
        return "ExceptionEvent"
