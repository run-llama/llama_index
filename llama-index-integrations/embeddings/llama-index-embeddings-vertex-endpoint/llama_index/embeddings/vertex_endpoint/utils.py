import abc
from typing import List, Any, Dict


class BaseIOHandler(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return (
            hasattr(subclass, "serialize_input")
            and callable(subclass.serialize_input)
            and hasattr(subclass, "deserialize_output")
            and callable(subclass.deserialize_output)
            or NotImplemented
        )

    @abc.abstractmethod
    def serialize_input(self, request: List[str]) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize_output(self, response: Any) -> List[List[float]]:
        raise NotImplementedError


class IOHandler(BaseIOHandler):
    """Handles serialization of input and deserialization of output."""

    def serialize_input(self, request: List[str]) -> List[Dict[str, Any]]:
        return [{"inputs": text} for text in request]

    def deserialize_output(self, response: Any) -> List[List[float]]:
        return [prediction[0] for prediction in response.predictions]
