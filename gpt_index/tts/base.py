"""Text to speech module."""
from abc import ABC, abstractmethod


class BaseTTS(ABC):
    """Base class for text to speech modules."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_audio(self, text: str) -> None:
        raise NotImplementedError(
            "generate_audio method should be implemented by subclasses"
        )
