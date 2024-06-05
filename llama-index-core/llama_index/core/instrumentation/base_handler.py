from abc import ABC, abstractmethod


class BaseInstrumentationHandler(ABC):
    @classmethod
    @abstractmethod
    def init(cls):
        """Initialize the instrumentation handler."""
