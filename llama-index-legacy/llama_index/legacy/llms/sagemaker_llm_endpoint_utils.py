import abc
import codecs
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botocore.response import StreamingBody

from llama_index.legacy.bridge.pydantic import BaseModel, Field


class BaseIOHandler(BaseModel, metaclass=abc.ABCMeta):
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
            and hasattr(subclass, "deserialize_streaming_output")
            and callable(subclass.deserialize_streaming_output)
            and hasattr(subclass, "remove_prefix")
            and callable(subclass.remove_prefix)
            or NotImplemented
        )

    @abc.abstractmethod
    def serialize_input(self, request: str, model_kwargs: dict) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize_output(self, response: "StreamingBody") -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize_streaming_output(self, response: bytes) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def remove_prefix(self, response: str, prompt: str) -> str:
        raise NotImplementedError


class IOHandler(BaseIOHandler):
    content_type: str = "application/json"
    accept: str = "application/json"

    def serialize_input(self, request: str, model_kwargs: dict) -> bytes:
        request_str = json.dumps({"inputs": request, "parameters": model_kwargs})
        return request_str.encode("utf-8")

    def deserialize_output(self, response: "StreamingBody") -> str:
        return json.load(codecs.getreader("utf-8")(response))[0]["generated_text"]

    def deserialize_streaming_output(self, response: bytes) -> str:
        response_str = (
            response.decode("utf-8").lstrip('[{"generated_text":"').rstrip('"}]')
        )
        clean_response = '{"response":"' + response_str + '"}'

        return json.loads(clean_response)["response"]

    def remove_prefix(self, raw_text: str, prompt: str) -> str:
        return raw_text[len(prompt) :]
