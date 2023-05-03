"""Text to speech module."""
from abc import ABC, abstractmethod
from typing import Any


class BaseTTS(ABC):
    """Base class for text to speech modules."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_audio(self, text: str) -> Any:
        """Generate audio from text.

        NOTE: return type is Any, but it should be any object that can be fed
        as `data` into IPython.display.Audio(). This includes numpy array, list,
        unicode, str or bytes

        """
        raise NotImplementedError(
            "generate_audio method should be implemented by subclasses"
        )
