import abc
import json
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from botocore.response import StreamingBody

from llama_index.legacy.bridge.pydantic import Field


class BaseIOHandler(metaclass=abc.ABCMeta):
    content_type: str = Field(
        description="The MIME type of the input data in the request body.",
    )
    accept: str = Field(
        description="The desired MIME type of the inference response from the model container.",
    )

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return (
            hasattr(subclass, "content_type")
            and hasattr(subclass, "accept")
            and hasattr(subclass, "serialize_input")
            and callable(subclass.serialize_input)
            and hasattr(subclass, "deserialize_output")
            and callable(subclass.deserialize_output)
            or NotImplemented
        )

    @abc.abstractmethod
    def serialize_input(self, request: List[str], model_kwargs: dict) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize_output(self, response: "StreamingBody") -> List[List[float]]:
        raise NotImplementedError


class IOHandler(BaseIOHandler):
    content_type: str = "application/json"
    accept: str = "application/json"

    def serialize_input(self, request: List[str], model_kwargs: dict) -> bytes:
        request_str = json.dumps({"inputs": request, **model_kwargs})
        return request_str.encode("utf-8")

    def deserialize_output(self, response: "StreamingBody") -> List[List[float]]:
        response_json = json.loads(response.read().decode("utf-8"))
        return response_json["vectors"]
