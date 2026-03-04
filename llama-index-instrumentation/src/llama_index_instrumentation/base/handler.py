from abc import ABC, abstractmethod


class BaseInstrumentationHandler(ABC):
    @classmethod
    @abstractmethod
    def init(cls) -> None:
        """Initialize the instrumentation handler."""
