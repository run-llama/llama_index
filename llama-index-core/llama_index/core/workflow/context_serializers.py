import base64
import json
import pickle
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

from llama_index.core.schema import BaseComponent
from .utils import import_module_from_qualified_name, get_qualified_name


class BaseSerializer(ABC):
    @abstractmethod
    def serialize(self, value: Any) -> str:
        ...

    @abstractmethod
    def deserialize(self, value: str) -> Any:
        ...


class JsonSerializer(BaseSerializer):
    def serialize(self, value: Any) -> str:
        if isinstance(value, BaseComponent):
            return json.dumps(
                {
                    "__is_component": True,
                    "value": value.to_dict(),
                    "qualified_name": get_qualified_name(value),
                }
            )
        elif isinstance(value, BaseModel):
            return json.dumps(
                {
                    "__is_pydantic": True,
                    "value": value.model_dump(),
                    "qualified_name": get_qualified_name(value),
                }
            )

        return json.dumps(value)

    def deserialize(self, value: str) -> Any:
        data = json.loads(value)

        if (
            isinstance(data, dict)
            and data.get("__is_pydantic")
            and data.get("qualified_name")
        ):
            module_class = import_module_from_qualified_name(data["qualified_name"])
            return module_class.model_validate(data["value"])
        elif (
            isinstance(data, dict)
            and data.get("__is_component")
            and data.get("qualified_name")
        ):
            module_class = import_module_from_qualified_name(data["qualified_name"])
            return module_class.from_dict(data["value"])

        return data


class JsonPickleSerializer(JsonSerializer):
    def serialize(self, value: Any) -> str:
        """Serialize while prioritizing JSON, falling back to Pickle."""
        try:
            return super().serialize(value)
        except Exception:
            return base64.b64encode(pickle.dumps(value)).decode("utf-8")

    def deserialize(self, value: str) -> Any:
        """Deserialize while prioritizing Pickle, falling back to JSON."""
        try:
            return pickle.loads(base64.b64decode(value))
        except Exception:
            return super().deserialize(value)
