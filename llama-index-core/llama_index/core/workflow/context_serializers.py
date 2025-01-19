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
    def _serialize_value(self, value: Any) -> Any:
        """Helper to serialize a single value."""
        if isinstance(value, BaseComponent):
            return {
                "__is_component": True,
                "value": value.to_dict(),
                "qualified_name": get_qualified_name(value),
            }
        elif isinstance(value, BaseModel):
            return {
                "__is_pydantic": True,
                "value": value.model_dump(),
                "qualified_name": get_qualified_name(value),
            }
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        return value

    def serialize(self, value: Any) -> str:
        try:
            serialized_value = self._serialize_value(value)
            return json.dumps(serialized_value)
        except Exception as e:
            raise ValueError(f"Failed to serialize value: {type(value)}: {value!s}")

    def _deserialize_value(self, data: Any) -> Any:
        """Helper to deserialize a single value."""
        if isinstance(data, dict):
            if data.get("__is_pydantic") and data.get("qualified_name"):
                module_class = import_module_from_qualified_name(data["qualified_name"])
                return module_class.model_validate(data["value"])
            elif data.get("__is_component") and data.get("qualified_name"):
                module_class = import_module_from_qualified_name(data["qualified_name"])
                return module_class.from_dict(data["value"])
            return {k: self._deserialize_value(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_value(item) for item in data]
        return data

    def deserialize(self, value: str) -> Any:
        data = json.loads(value)
        return self._deserialize_value(data)


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
