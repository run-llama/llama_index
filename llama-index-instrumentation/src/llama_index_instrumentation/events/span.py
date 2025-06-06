from .base import BaseEvent


class SpanDropEvent(BaseEvent):
    """
    SpanDropEvent.

    Args:
        err_str (str): Error string.

    """

    err_str: str

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SpanDropEvent"
